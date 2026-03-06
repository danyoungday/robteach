from robosuite_env import RobosuiteGymEnv


class CurriculumEnv(RobosuiteGymEnv):
    def __init__(self, robots, **kwargs):
        super().__init__(env_name="Lift", robots=robots, **kwargs)
