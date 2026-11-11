import json
import numpy as np
import tensorflow as tf
from utils import *
from gae import GAE
from obs_norm import FixedObservationScaler

class TRPO():
    def __init__(self, args, env, sess):
        self.args = args
        self.sess = sess
        self.env = env

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

    def build_model(self):
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        # Shared running mean/std normalizer for theta_dot ONLY (policy and value net);
        # cos/sin(theta) pass through. Statistics stay frozen during an iteration and are
        # refreshed at its end (see train()).
        self.obs_norm = FixedObservationScaler(self.observation_size)
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])

        #Mean of old action distribution
        self.old_action_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.old_action_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

        #NN framework for action distribution
        self.action_dist_mu, action_dist_logstd = self.build_policy(self.obs)

        #Construct distribution by repeating action_dis_logstd
        self.action_dist_logstd = tf.tile(action_dist_logstd, (tf.shape(self.obs)[0],1))

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
        
        # Get trainable variables for the policy (NN weights)
        tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')
        for i in tr_vrbs:
            print(i.op.name)
        
        # Maximize surrogate function over policy parameter 'theta' represented by neural network weights
        self.pg = FLAT_GRAD(surr_single_state, tr_vrbs)
        
        #KL divergence where first argument is fixed
        kl_first_fixed = GAUSS_KL_FIRST_FIX(self.action_dist_mu, self.action_dist_logstd) / batch_size
        
        #Gradient of KL divergence w.r.t. theta (NN policy weights)
        first_kl_grads = tf.gradients(kl_first_fixed, tr_vrbs)
        
        '''
        REVIEW FROM HERE ONWARDS
        #??????????????????????????????????????????????????????????
        '''
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
        
        self.sess.run(tf.global_variables_initializer())
        
        
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
        action = np.squeeze(np.concatenate([each_path["Action"] for each_path in batch_path]))
        
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
        # Refresh normalizer only now: sampling, advantages, policy step and value fit
        # of this iteration all used the same statistics.
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
        logs["Episode_Avg_Reward"] = logs["Total Sum"] / logs["Num episode"]
        logs["Final_Action"] = np.squeeze(np.concatenate([each_path["Action"] for each_path in batch_path]))
        logs["Observation"] = np.squeeze(np.concatenate([each_path["Observation"] for each_path in batch_path]))
        logs["Reward"] = np.squeeze(np.concatenate([each_path["Reward"] for each_path in batch_path]))
        return logs
        

    #USE SOFTMAX RELU INSTEAD OF RELU, OUTPUT WEIGHTS/BIASES IN EASIER FORMAT   
    #Set up NN to parameterize the control policy
    def build_policy(self, states, name='Policy'):
        print('Initializing Policy network')
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            h1 = LINEAR(self.obs_norm(states), self.args.hidden_size, name='h1')
            h1_n1 = tf.nn.relu(h1)
            #h1_n1 = tf.nn.softmax(h1)
            h2 = LINEAR(h1_n1, self.args.hidden_size, name='h2')
            h2_n1 = tf.nn.relu(h2)
            #h2_n1 = tf.nn.softmax(h2)
            h3 = LINEAR(h2_n1, self.action_size, name='h3')

            #Initialize action std_deviation to 1 (logstd=0), same as TRPO-CBF
            #(author code: 0.01*randn, i.e. logstd ~ 0)
            # Initial action std in torque units (action range is +-15): log(3) by default.
            action_dist_logstd = tf.get_variable('logstd', initializer=tf.constant_initializer(getattr(self.args, 'init_logstd', 0.)), shape=[1, self.action_size])
            
        return h3, action_dist_logstd
    
    def act(self, obs):
        #Need to expand first dimension (batch axis), make [1, observation size]
        obs_expanded = np.expand_dims(np.squeeze(obs), 0)
        #obs_expanded = obs
        #Get action distribution from policy network
        action_dist_mu, action_dist_logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs:obs_expanded})
        #Sample action from gaussian distribution
        action = np.random.normal(loc=action_dist_mu, scale=np.exp(action_dist_logstd))
        return action, action_dist_mu, action_dist_logstd
    
    def rollout(self):
        paths, timesteps = [], 0
        while timesteps < self.args.timesteps_per_batch:
            obs, next_obs, actions, rewards, means, logstds, done = [], [], [], [], [], [], []
            state = self.env.reset()
            for t in range(self.args.max_path_length):
                before = np.asarray(state).copy()
                raw, mean, logstd = self.act(before)
                action = np.asarray(raw).reshape(self.action_size)
                after, reward, terminal, info = self.env.step(action)
                end = terminal or t + 1 == self.args.max_path_length
                obs.append(before)
                next_obs.append(np.asarray(after).copy())
                actions.append(action)
                means.append(np.asarray(mean).reshape(self.action_size))
                logstds.append(np.asarray(logstd).reshape(self.action_size))
                rewards.append(float(reward))
                done.append(end)
                state = after
                if end:
                    break
            paths.append(dict(Observation=np.asarray(obs), NextObservation=np.asarray(next_obs),
                Action=np.asarray(actions), Action_mu=np.asarray(means),
                Action_logstd=np.asarray(logstds), Reward=np.asarray(rewards), Done=np.asarray(done)))
            timesteps += len(rewards)
            self.ep_count = getattr(self, 'ep_count', 0) + 1
            _o = np.concatenate([np.asarray(obs), np.asarray(next_obs)])
            print('LIVE ' + json.dumps(dict(episode=self.ep_count, reward=float(np.sum(rewards)),
                  max_angle=float(np.max(np.abs(np.arctan2(_o[:, 1], _o[:, 0])))))), flush=True)
        self.num_epi = len(paths)
        return paths

    def sim(self):
        observation = self.env.reset()
        total = 0
        for t in range(200):
            #Render environment
            self.env.render()
            
            #Get action from NN policy
            obs_expanded = np.expand_dims(np.squeeze(observation), 0)
            #obs_expanded = obs
            #Get action distribution from policy network
            action_dist_mu, action_dist_logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs:obs_expanded})
            #Sample action from gaussian distribution
            action = np.random.normal(loc=action_dist_mu, scale=np.exp(action_dist_logstd))
            
            observation, reward, done, info = self.env.step(np.asarray(action).reshape(self.action_size))
            total = total + reward
            if done:
                print("Accumulated Reward: {}".format(total))
                break                
