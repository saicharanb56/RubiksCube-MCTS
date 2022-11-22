from rubikscube import Cube, HalfTurnMetric, QuarterTurnMetric, Metric
import gym
import numpy as np

from gym import spaces

from copy import deepcopy

from typing import Tuple, Optional, List


class CubeEnv(gym.Env):
    def __init__(self, metric: str):

        super(CubeEnv, self).__init__()

        assert metric in ['half-turn', 'quarter-turn']
        self.metric: Metric = HalfTurnMetric if metric == 'half-turn' else QuarterTurnMetric
        self.action_space: spaces.Discrete = spaces.Discrete(
            self.metric.to_int())
        self.observation_space: spaces.MultiBinary = spaces.MultiBinary(480)
        self.reward_range: spaces.Discrete = spaces.Discrete(2)

        self._cube: Optional[Cube] = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:

        assert self.action_space.contains(
            action), f"{action!r} ({type(action)}) invalid"
        assert self._cube is not None, "Call reset before using step method."

        self._cube.turn(action)
        done: bool = self._cube.solved()
        return self._cube.representation(), int(done), done, {}

    def reset(self, scramble_moves: int = 100) -> Tuple[np.ndarray, dict]:

        self._cube = Cube.cube_htm(
        ) if self.metric == HalfTurnMetric else Cube.cube_qtm()
        self._cube.scramble(scramble_moves)
        return self._cube.representation(), {}

    def get_state(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        assert self._cube is not None, "Call reset before using step method."
        return deepcopy(self._cube.get_state())

    def set_state(self, state: Tuple[List[int], List[int], List[int],
                                     List[int]]):
        assert self._cube is not None, "Call reset before using step method."
        self._cube.set_state(deepcopy(state))

    def render(self):
        print(self._cube)
