import ray
import torch.nn as nn

from typing import Tuple, List

from env import CubeEnv
import numpy as np
import gym


@ray.remote
class Scrambler:
    def __init__(
        self,
        nn_e: nn.Module,
        env: gym.Env = CubeEnv,
    ):
        self._env = env
        self.nn_e = nn_e

    def sample_once(self,
                    n_scramble: int = 50,
                    *args,
                    **kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        self.env = self._env(*args, **kwargs)
        s, _ = self.env.reset(n_scramble)

        env_state = self.env.get_state()

        s_children = []

        for action in range(self.env.action_space.start,
                            self.env.action_space.n):
            s_child, *_ = self.env.step(action)
            s_children.append(s_child)
            self.env.set_state(env_state)

        v = self.nn_e(s_children)

        return (s, v)


@ray.remote
def train():
    pass


if __name__ == "__main__":
    pass
