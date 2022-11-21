import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env import CubeEnv
import unittest


class TestCubeEnv(unittest.TestCase):
    def setUp(self):
        self.env_qtm = CubeEnv(metric='quarter-turn')
        self.env_htm = CubeEnv(metric='half-turn')

    def test_htm_reset(self):
        s, i = self.env_htm.reset(100)

        self.assertEqual(i, {})
        self.assertEqual(s.shape, (480, ))

    def test_qtm_reset(self):
        s, i = self.env_qtm.reset(100)

        self.assertEqual(i, {})
        self.assertEqual(s.shape, (480, ))

    def test_htm_step(self):
        self.env_htm.reset()

        action = self.env_htm.action_space.sample()
        s, r, d, i = self.env_htm.step(action)

        self.assertEqual(i, {})
        self.assertEqual(s.shape, (480, ))
        self.assertIn(r, [0., 1.])
        self.assertIn(d, [True, False])

    def test_qtm_step(self):
        self.env_qtm.reset()

        action = self.env_qtm.action_space.sample()
        s, r, d, i = self.env_qtm.step(action)

        self.assertEqual(i, {})
        self.assertEqual(s.shape, (480, ))
        self.assertIn(r, [0., 1.])
        self.assertIn(d, [True, False])

    def test_htm_get_set(self):

        s, _ = self.env_htm.reset()

        state = self.env_htm.get_state()

        action = self.env_htm.action_space.sample()
        s, *_ = self.env_htm.step(action)

        another_env = CubeEnv('half-turn')
        another_env.reset()
        another_env.set_state(state=state)

        another_s, *_ = another_env.step(action)

        self.assertTrue((another_s == s).all(),
                        "State represenations differ after setting states")

    def test_qtm_get_set(self):

        s, _ = self.env_qtm.reset()

        state = self.env_qtm.get_state()

        action = self.env_qtm.action_space.sample()
        s, *_ = self.env_qtm.step(action)

        another_env = CubeEnv('quarter-turn')
        another_env.reset()
        another_env.set_state(state=state)

        another_s, *_ = another_env.step(action)

        self.assertTrue((another_s == s).all(),
                        "State represenations differ after setting states")


if __name__ == "__main__":
    unittest.main()
