import logging

import sys
import os

from rubikscube import Cube, HalfTurnMetric
import unittest
import timeit


class TestBenchMarkEnv(unittest.TestCase):
    def setUp(self):
        self.trials = int(1e7)
        self.log = logging.getLogger('BenchLogger')

    def test_turn_repr_solved(self):
        t_turn_repr_solve = timeit.timeit(
            'cube.turn(0);cube.representation();cube.solved()',
            setup='from rubikscube import Cube;cube=Cube.cube_htm()',
            number=self.trials)
        self.log.debug(
            f"time -- turn + repr + solved :::  {t_turn_repr_solve}")

    def test_turn_repr(self):
        t_turn_repr = timeit.timeit(
            'cube.turn(0);cube.representation()',
            setup='from rubikscube import Cube;cube=Cube.cube_htm()',
            number=self.trials)
        self.log.debug(f"time -- turn + repr  :::  {t_turn_repr}")

    def test_turn(self):
        t_turn = timeit.timeit(
            'cube.turn(0)',
            setup='from rubikscube import Cube;cube=Cube.cube_htm()',
            number=self.trials)

        self.log.debug(f"time -- turn  :::  {t_turn}")

    def test_env(self):
        t_env = timeit.timeit(
            'env.step(0)',
            setup=
            "from rubikscube import HalfTurnMetric;from env import CubeEnv;env=CubeEnv('half-turn');env.reset()",
            number=self.trials)

        self.log.debug(f"time -- env  :::  {t_env}")


if __name__ == "__main__":

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("BenchLogger").setLevel(logging.DEBUG)
    unittest.main()
