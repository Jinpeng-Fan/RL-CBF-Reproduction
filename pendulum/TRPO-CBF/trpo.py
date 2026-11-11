import json
import numpy as np
import tensorflow as tf
import gym
from fair_environment import configure_comparison_env
from utils import *
from gae import GAE
from barrier_comp import BARRIER
from obs_norm import FixedObservationScaler
import cbf
import dynamics_gp

class TRPO():
    def __init__(self, args, env, sess):
        self.firstIter = 1
        self.count = 1
        self.args = args
        if args.timesteps_per_batch < args.max_path_length:
            raise ValueError('timesteps_per_batch must cover at least one full episode.')
        self.enforce_speed_limit = getattr(args, 'enforce_speed_limit', False)
        self.sess = sess
        self.env = env
        self.torque_bound = 15.
        self.max_speed = 60.

        #Set up observation space and action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print('Observation space', self.observation_space)
        print('Action space', self.action_space)

        #Determine dimensions of observation & action space
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.action_space.shape[0]

        # Build neural network model for observations/actions
        self.build_model()
        
        # Build barrier function model
        cbf.build_barrier(self)
        
        # Build GP model of dynamics
        dynamics_gp.build_GP_model(self)

        # Pre-fit GP so episode 1 does not use the unfit prior (std=1)
        self.calibrate_GP()


    # Build RL policy improvement model based on TRPO
    def build_model(self):
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        # Shared running mean/std normalizer for theta_dot ONLY (policy, value net,
        # compensator); cos/sin(theta) pass through. Statistics stay frozen during an
        # iteration and are refreshed at its end (see train()).
        self.obs_norm = FixedObservationScaler(self.observation_size)
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])

        #Mean of old action distribution
        self.old_action_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.old_action_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

        #NN framework for action distribution
        self.action_dist_mu, action_dist_logstd = self.build_policy(self.obs)

        # Get trainable variables for the policy (NN weights)                                                     
        tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')
        for i in tr_vrbs:
            print(i.op.name)
        
        #Construct distribution by repeating action_dis_logstd
        self.action_dist_logstd = tf.tile(action_dist_logstd, (tf.shape(self.obs)[0], 1))

        #Probability of action under old policy vs. new policy
        self.log_policy = LOG_POLICY(self.action_dist_mu, self.action_dist_logstd, self.action)
        self.log_old_policy = LOG_POLICY(self.old_action_dist_mu, self.old_action_dist_logstd, self.action)
        policy_ratio = tf.exp(self.log_policy - self.log_old_policy)
        
        #Number of observations in batch
        batch_size = tf.cast(tf.shape(self.obs)[0], tf.float32)
        
        '''
        Equation (14) in paper
        Contribution of a single s_n : Expectation over a~q[ (new policy / q(is)) * advantage_old]
        '''
        surr_single_state = -tf.reduce_mean(policy_ratio*self.advantage)
        
        
        #Define KL divergence and shannon entropy, averaged over a set of inputs (policies)        
        kl = GAUSS_KL(self.old_action_dist_mu, self.old_action_dist_logstd, self.action_dist_mu, self.action_dist_logstd) / batch_size
        ent = GAUSS_ENTROPY(self.action_dist_mu, self.action_dist_logstd) / batch_size
        
        
        #Define 'loss' quantities to constrain or maximize
        self.losses = [surr_single_state, kl, ent]
        
        # Maximize surrogate function over policy parameter 'theta' represented by neural network weights
        self.pg = FLAT_GRAD(surr_single_state, tr_vrbs)
        
        #KL divergence where first argument is fixed
        kl_first_fixed = GAUSS_KL_FIRST_FIX(self.action_dist_mu, self.action_dist_logstd) / batch_size
        
        #Gradient of KL divergence w.r.t. theta (NN policy weights)
        first_kl_grads = tf.gradients(kl_first_fixed, tr_vrbs)
        
        self.flat_tangent = tf.placeholder(tf.float32,[None])
        tangent = list()
        start = 0
        for vrbs in tr_vrbs:
            variable_size = np.prod(vrbs.get_shape().as_list())
            param = tf.reshape(self.flat_tangent[start:(start+variable_size)], vrbs.get_shape())
            tangent.append(param)
            start += variable_size
        '''
            Gradient of KL with tangent vector
            gradient_w_tangent : list of KL_prime*y for each variables
        '''
        gradient_w_tangent = [tf.reduce_sum(kl_g*t) for (kl_g, t) in zip(first_kl_grads, tangent)]
        
        '''
			From derivative of KL_prime*y : [dKL/dx1, dKL/dx2...]*y
				y -> Ay, A is n by n matrix but hard to implement(numerically solving (n*n)*(n*1))
				so first multiply target 'y' to gradient and take derivation
		    'self.FVP'	Returns : [d2KL/dx1dx1+d2KL/dx1dx2..., d2KL/dx1dx2+d2KL/dx2dx2..., ...]*y
			So get (second derivative of KL divergence)*y for each variable => y->JMJy (Fisher Vector Product)
		'''
        self.FVP = FLAT_GRAD(gradient_w_tangent, tr_vrbs)
        
        #Get actual parameter value
        self.get_value = GetValue(self.sess, tr_vrbs, name='Policy')
        
        #Set parameter values
        self.set_value = SetValue(self.sess, tr_vrbs, name='Policy')
        
        #Estimate of the advantage function 
        self.gae = GAE(self.sess, self.observation_size, self.args.gamma, self.args.lamda, self.args.vf_constraint,
                       obs_normalizer=self.obs_norm,
                       vf_method=getattr(self.args, 'vf_method', 'adam'))

        #Intialization of the barrier function compensator
        self.bar_comp = BARRIER(self.args, self.sess, self.observation_size, self.action_size,
                                obs_normalizer=self.obs_norm)

        #Variable initializers
        self.sess.run(tf.global_variables_initializer())
        

    #Train TRPO policy 
    def train(self):
        batch_path = self.rollout()
        theta_prev = self.get_value()
        
        #Get advantage from gae (train value function NN)
        advantage_estimated = self.gae.get_advantage(batch_path)

        
        #Put all paths in batch in a numpy array to feed to network as [batch size, action/observation size]
        #Those batches come from OLD policy before updating theta
        action_dist_mu = np.squeeze(np.concatenate([each_path["Action_mu"] for each_path in batch_path]))
        action_dist_logstd = np.squeeze(np.concatenate([each_path["Action_logstd"] for each_path in batch_path]))
        observation = np.squeeze(np.concatenate([each_path["Observation"] for each_path in batch_path]))
        action = np.concatenate([p["Action_RL"] for p in batch_path]).reshape(-1)

        
        
        #Obtain policy gradient of advantage function w.r.t. theta (g in paper)
        feed_dict = {self.obs:observation, self.action:np.expand_dims(action, axis=1), self.advantage:advantage_estimated, self.old_action_dist_mu:np.expand_dims(action_dist_mu, axis=1), self.old_action_dist_logstd:np.expand_dims(action_dist_logstd, axis=1)}
        #feed_dict = {self.obs:observation, self.action:action, self.advantage:advantage_estimated, self.old_action_dist_mu:action_dist_mu, self.old_action_dist_logstd:action_dist_logstd}
        policy_g = self.sess.run(self.pg, feed_dict=feed_dict)
        
        # Computing fisher vector product : FIM * (policy gradient) where FIM = Fisher Information Matrix
        def fisher_vector_product(gradient):
            feed_dict[self.flat_tangent] = gradient
            return self.sess.run(self.FVP, feed_dict=feed_dict)
            
        def evaluate_policy(theta):
            self.set_value(theta)
            loss, kl = self.sess.run(self.losses[:2], feed_dict=feed_dict)
            return float(loss), float(kl)

        new_theta, self.last_policy_update = TRUST_REGION_STEP(
            evaluate_policy, theta_prev, policy_g, fisher_vector_product,
            self.args.kl_constraint, self.args.num_backtracking)
        self.set_value(new_theta, update_info=0)

        #Update value function neural network
        #Policy update is performed using old value function parameter
        self.gae.train()
        
        #After update, store values at log
        surrogate_after, kl_after, _ = self.sess.run(self.losses, feed_dict=feed_dict)
        logs = {"Surrogate loss":surrogate_after, "KL_DIV":kl_after}
        value_info = self.gae.last_value_update
        logs['Value_update_accepted'] = value_info['accepted']
        logs['Value_gradient_fallback'] = value_info['method'] == 'gradient_fallback'
        logs['Value_loss_before'] = value_info['loss_before']
        logs['Value_loss_after'] = value_info['loss_after']
        logs['Value_backtracks'] = value_info['backtracks']
        logs['Value_update_status'] = value_info['status']
        logs['Value_explained_variance_before'] = value_info['explained_variance_before']
        logs['Value_explained_variance_after'] = value_info['explained_variance_after']
        logs['Policy_update_accepted'] = self.last_policy_update['accepted']
        logs['Policy_backtracks'] = self.last_policy_update['backtracks']
        logs['Policy_step_status'] = self.last_policy_update['status']
        logs["Total Step"] = sum([len(path["Reward"]) for path in batch_path])
        logs["Num episode"] = len([path["Reward"] for path in batch_path])
        logs["Total Sum"] = sum([sum(path["Reward"]) for path in batch_path])
        logs["Episode Avg. Reward"] = logs["Total Sum"] / logs["Num episode"]
        # Refresh normalizer first, so the compensator is fitted under the scaling it
        # will be deployed with (Algorithm 1: policy update, then compensator fit).
        # Fit the compensator on this iteration's whole batch, after the policy update.
        self.bar_comp.get_training_rollouts(batch_path)
        barr_loss = self.bar_comp.train()
        # Iteration k=0 deploys no compensator (Algorithm 1, lines 2-9).
        self.firstIter = 0
        logs["Compensator_Fit"] = barr_loss
        logs['Compensator_accepted_steps'] = self.bar_comp.last_fit['accepted_steps']
        logs['Compensator_loss_before'] = self.bar_comp.last_fit['loss_before']
        logs['Compensator_loss_after'] = self.bar_comp.last_fit['loss_after']
        logs['Compensator_stop_reason'] = self.bar_comp.last_fit['stop_reason']
        logs['GP_angle_bound_exceeded_fraction'] = float(np.mean(np.concatenate([
            p['GP_angle_bound_exceeded'] for p in batch_path])))
        logs['CBF_min_relaxed_margin'] = min(float(np.min(p['CBF_relaxed_margin'])) for p in batch_path)
        logs["GP_samples"] = len(self.gp_X)
        logs["Max_CBF_slack"] = max(float(np.max(p['CBF_slack'])) for p in batch_path)
        logs["Final_Action"] = np.squeeze(np.concatenate([each_path["Action"] for each_path in batch_path]))
        logs["Action_bar"] = np.squeeze(np.concatenate([each_path["Action_bar"] for each_path in batch_path]))
        logs["Action_BAR"] = np.squeeze(np.concatenate([each_path["Action_BAR"] for each_path in batch_path]))
        logs["Observation"] = np.squeeze(np.concatenate([each_path["Observation"] for each_path in batch_path]))
        logs["Reward"] = np.squeeze(np.concatenate([each_path["Reward"] for each_path in batch_path]))
        return logs
        
        
    #Set up NN to parameterize the control policy
    def build_policy(self, states, name='Policy'):
        print('Initializing Policy network')
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            h1 = LINEAR(self.obs_norm(states), self.args.hidden_size, name='h1')
            h1_n1 = tf.nn.relu(h1)
            h2 = LINEAR(h1_n1, self.args.hidden_size, name='h2')
            h2_n1 = tf.nn.relu(h2)
            h3 = LINEAR(h2_n1, self.action_size, name='h3')

            #Initialize action std_deviation
            #init = lambda shape, dtype, partition_info=None : 0.01*np.random.randn(*shape)
            #action_dist_logstd = tf.get_variable('logstd', initializer=init, shape=[1, self.action_size])
            
            #Initialize action std_deviation to 1 (logstd=0), same as the TRPO baseline
            # Initial action std in torque units (action range is +-15): log(3) by default.
            action_dist_logstd = tf.get_variable('logstd', initializer=tf.constant_initializer(getattr(self.args, 'init_logstd', 0.)), shape=[1, self.action_size])
            
        return h3, action_dist_logstd
    
    #Get action from the current observation (sampled based on NN policy)
    def act(self, obs):
        #Need to expand first dimension (batch axis), make [1, observation size]
        obs_expanded = np.expand_dims(np.squeeze(obs), 0)
        #obs_expanded = obs
        #Get action distribution from policy network
        action_dist_mu, action_dist_logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs:obs_expanded})
        #Sample action from gaussian distribution
        action = np.random.normal(loc=action_dist_mu, scale=np.exp(action_dist_logstd))
        return action, action_dist_mu, action_dist_logstd


    #Simulate dynamics for a given rollout
    def rollout(self):
        """Collect timesteps_per_batch steps (whole episodes) per policy
        iteration; the GP is refit after each episode (Algorithm 1, line 22)."""
        paths = []
        timesteps = 0
        while timesteps < self.args.timesteps_per_batch:
            paths.append(self.rollout_episode())
            timesteps += len(paths[-1]['Reward'])
        self.num_epi = len(paths)
        return paths

    def rollout_episode(self):
        """One episode; sample BEFORE the final filter."""
        obs, next_obs, actions, raw_actions, rewards = [], [], [], [], []
        means, logstds, bars, comps, done, slacks = [], [], [], [], [], []
        diagnostics = []
        state = self.env.reset()
        while abs(self.env.unwrapped.state[0]) > 1.0:
            state = self.env.reset()
        for t in range(self.args.max_path_length):
            before = np.asarray(state).copy()
            raw, mean, logstd = self.act(before)
            comp = (np.zeros_like(raw) if self.firstIter else
                    self.bar_comp.get_action(before))
            candidate = raw + comp
            f, g, x, Sigma = dynamics_gp.get_GP_dynamics(self, before)
            bar = cbf.control_barrier(self, before, candidate, f, g, x, Sigma)
            executed = np.asarray(candidate + bar).reshape(self.action_size)
            after, reward, terminal, info = self.env.step(executed)
            actual = dynamics_gp.state_from_obs(after)
            error = actual - self.last_cbf['predicted_next_state']
            error[0] = (error[0]+np.pi) % (2*np.pi)-np.pi
            bound = self.barrier_kd*self.last_cbf['sigma']
            diagnostics.append(dict(
                predicted=self.last_cbf['predicted_next_state'].copy(),
                error=error.copy(), bound=bound.copy(),
                angle_bound_exceeded=bool(abs(error[0]) > bound[0]+1e-8),
                unrelaxed_margin=self.last_cbf['unrelaxed_margin'],
                relaxed_margin=self.last_cbf['min_robust_margin']))
            end = terminal or t + 1 == self.args.max_path_length
            obs.append(before)
            next_obs.append(np.asarray(after).copy())
            actions.append(executed)
            raw_actions.append(np.asarray(raw).reshape(self.action_size))
            means.append(np.asarray(mean).reshape(self.action_size))
            logstds.append(np.asarray(logstd).reshape(self.action_size))
            bars.append(np.asarray(bar).reshape(self.action_size))
            comps.append(np.asarray(comp).reshape(self.action_size))
            rewards.append(reward)
            done.append(end)
            slacks.append(self.last_cbf['slack'])
            state = after
            if end:
                break
        path = dict(Observation=np.asarray(obs), NextObservation=np.asarray(next_obs),
                    Action=np.asarray(actions), Action_RL=np.asarray(raw_actions),
                    Action_mu=np.asarray(means), Action_logstd=np.asarray(logstds),
                    Action_bar=np.asarray(bars), Action_BAR=np.asarray(comps),
                    Reward=np.asarray(rewards), Done=np.asarray(done),
                    CBF_slack=np.asarray(slacks))
        path['GP_predicted_next_state'] = np.asarray([d['predicted'] for d in diagnostics])
        path['GP_prediction_error'] = np.asarray([d['error'] for d in diagnostics])
        path['GP_confidence_radius'] = np.asarray([d['bound'] for d in diagnostics])
        path['GP_angle_bound_exceeded'] = np.asarray([d['angle_bound_exceeded'] for d in diagnostics])
        path['CBF_unrelaxed_margin'] = np.asarray([d['unrelaxed_margin'] for d in diagnostics])
        path['CBF_relaxed_margin'] = np.asarray([d['relaxed_margin'] for d in diagnostics])
        self.last_training_path = path
        self.ep_count = getattr(self, 'ep_count', 0) + 1
        _o = np.concatenate([path['Observation'], path['NextObservation']])
        print('LIVE ' + json.dumps(dict(episode=self.ep_count, reward=float(np.sum(path['Reward'])),
              max_angle=float(np.max(np.abs(np.arctan2(_o[:, 1], _o[:, 0])))),
              slack=float(np.max(path['CBF_slack'])),
              correction_rms=float(np.sqrt(np.mean(np.square(path['Action_bar'])))))), flush=True)
        dynamics_gp.update_GP_dynamics(self, path)
        self.count += 1
        return path


    def calibrate_GP(self):
        """Fit the GP on a short nominal-CBF rollout before learning starts.

        With the unfit prior (std=1), kd*std=1.5 rad exceeds the one-step
        reach 15*g[0]~0.014 rad, so every CBF constraint is infeasible and
        the filter chatters at +/-15; the compensator then imitates it.
        A separate env instance keeps the training initial-state RNG intact.
        """
        steps = getattr(self.args, 'gp_calibration_steps', 0)
        if steps <= 0:
            return
        rng = np.random.RandomState(getattr(self.args, 'random_seed', 1234))
        torque_std = getattr(self.args, 'calibration_torque_std', 1.0)
        env = configure_comparison_env(gym.make(self.env.spec.id))
        env.seed(getattr(self.args, 'random_seed', 1234))
        state = env.reset()
        obs, next_obs, actions = [], [], []
        for t in range(steps):
            before = np.asarray(state).copy()
            candidate = rng.normal(scale=torque_std, size=self.action_size)
            # Nominal model only (GP mean is zero before fitting), no robust margin
            f, g, x = dynamics_gp.get_dynamics(self, before, 0.0)
            bar = cbf.control_barrier(self, before, candidate, f, g, x, np.zeros((2, 2, 2)))
            executed = np.asarray(candidate + bar).reshape(self.action_size)
            after, _, terminal, _ = env.step(executed)
            obs.append(before)
            next_obs.append(np.asarray(after).copy())
            actions.append(executed)
            state = after
            if terminal:
                break
        env.close()
        dynamics_gp.update_GP_dynamics(self, dict(
            Observation=np.asarray(obs), NextObservation=np.asarray(next_obs),
            Action=np.asarray(actions)))
        angles = [dynamics_gp.state_from_obs(o)[0] for o in obs + next_obs[-1:]]
        self.calibration_info = dict(steps=len(obs), max_angle=float(np.max(np.abs(angles))))
        print('GP calibration: %d steps, max |theta| = %.3f' % (
            len(obs), self.calibration_info['max_angle']))

    #Simulate/Visualize latest policy
    def sim(self):
        observation = self.env.reset()
        while (self.env.unwrapped.state[0] > 1 or self.env.unwrapped.state[0] < -1):
                observation = self.env.reset()

        total = 0
        for t in range(200):
            #Render environment
            self.env.render()
            
            #Get action from NN policy
            obs_expanded = np.expand_dims(np.squeeze(observation), 0)
            
            #Get action distribution from policy network
            action_dist_mu, action_dist_logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs:obs_expanded})

            #Sample action from gaussian distribution
            action_rl = np.random.normal(loc=action_dist_mu, scale=np.exp(action_dist_logstd))

            #Get compensatory barrier action
            u_BAR_ = self.bar_comp.get_action(obs_expanded)
            u_RL = action_rl + u_BAR_
            
            #Compensate with barrier-based control
            [f,g,x,Sigma] = dynamics_gp.get_GP_dynamics(self,obs_expanded,u_RL)
            u_bar = cbf.control_barrier(self, obs_expanded, u_RL, f, g, x, Sigma)
            action = np.asarray(u_bar + u_RL).reshape(self.action_size)
            

            
            observation, reward, done, info = self.env.step(action)
            total = total + reward
            if done:
                print("Accumulated Reward: {}".format(total))
                break                
