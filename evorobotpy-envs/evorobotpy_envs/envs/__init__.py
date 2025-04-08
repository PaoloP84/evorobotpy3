from gymnasium.envs.registration import register

register(
    id='evorobotpy_envs/SlimeVolley-v0',
    entry_point='evorobotpy_envs.envs.slimevolley:SlimeVolleyEnv'
)

register(
    id='evorobotpy_envs/SlimeVolleyPixel-v0',
    entry_point='evorobotpy_envs.envs.slimevolley:SlimeVolleyPixelEnv'
)

register(
    id='evorobotpy_envs/SlimeVolleyNoFrameskip-v0',
    entry_point='evorobotpy_envs.envs.slimevolley:SlimeVolleyAtariEnv'
)

register(
    id='evorobotpy_envs/SlimeVolleySurvivalNoFrameskip-v0',
    entry_point='evorobotpy_envs.envs.slimevolley:SlimeVolleySurvivalAtariEnv'
)

register(
    id='evorobotpy_envs/SlimeVolleyHard-v0',
    entry_point='evorobotpy_envs.envs.slimevolleyhard:SlimeVolleyHardEnv'
)

register(
    id='evorobotpy_envs/SlimeVolleyHardLargeField-v0',
    entry_point='evorobotpy_envs.envs.slimevolleyhard:SlimeVolleyHardLargeFieldEnv'
)

register(
    id='evorobotpy_envs/GridWorld-v0',
    entry_point="evorobotpy_envs.envs.gridworld:GridWorldEnv",
    max_episode_steps=300,
)

register(
    id='evorobotpy_envs/GameOfLife-v0',
    entry_point="evorobotpy_envs.envs.gameoflife:GameOfLifeEnv",
)

register(
    id='evorobotpy_envs/DoublePole-v0',
    entry_point="evorobotpy_envs.envs.doublepole:DoublePoleEnv",
)
