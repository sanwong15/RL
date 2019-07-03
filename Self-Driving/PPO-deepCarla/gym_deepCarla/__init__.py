from gym.envs.registration import register

# TODO: Move this into the deepdrive package

register(
    id='DeepCarla-v0',
    entry_point='gym_deepCarla.envs.deepCarla_gym_env:DeepCarlaEnv',
    kwargs=dict(
        preprocess_with_tensorflow=False,
    ),
)

register(
    id='DeepCarlaPreproTensorflow-v0',
    entry_point='gym_deepCarla.envs.deepCarla_gym_env:DeepCarlaEnv',
    kwargs=dict(
        preprocess_with_tensorflow=True,
    ),
)
