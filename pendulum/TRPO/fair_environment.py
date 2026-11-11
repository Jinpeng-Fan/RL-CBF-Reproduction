"""Identical comparison environment; initial states use a separate RNG."""
import gym
import numpy as np

INITIAL_STATE_SEED = 1234
EPISODE_STEPS = 200

def configure_comparison_env(env, seed=None):
    base = env.unwrapped
    base.g, base.m, base.l, base.dt = 10.0, 1.0, 1.0, 0.05
    base.max_torque, base.max_speed = 15.0, 60.0
    base.action_space = gym.spaces.Box(low=-15., high=15., shape=(1,), dtype=np.float32)
    high = np.array([1., 1., 60.], dtype=np.float32)
    base.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
    env.action_space, env.observation_space = base.action_space, base.observation_space
    env._max_episode_steps = EPISODE_STEPS
    # Initial states depend on the run seed (same seed -> same states in all four algorithms).
    initial_rng = np.random.RandomState(INITIAL_STATE_SEED + (0 if seed is None else int(seed)))
    original_reset = env.reset

    def reset(*args, **kwargs):
        original_reset(*args, **kwargs)  # Reset Gym TimeLimit and rendering state.
        base.state = initial_rng.uniform(low=[-1., -1.], high=[1., 1.])
        base.last_u = None
        return base._get_obs()

    env.reset = reset
    return env
