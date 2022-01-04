from gym import ObservationWrapper


class AntWrapper(ObservationWrapper):
    def __init__(self, env):
        super(AntWrapper, self).__init__(env)
        self.observation_space.shape = (27,)

    def observation(self, observation):
        return observation[0:27]