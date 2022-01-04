from gym.envs.registration import register

register(
    id='HalfCheetahSimple-v0',
    entry_point='policygrad.environments.half_cheetah_simple:HalfCheetahEnv2',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)