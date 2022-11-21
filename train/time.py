from scrambler import Scrambler
from env import CubeEnv

import time

import ray

ray.init()

s = Scrambler.remote(lambda x: 1., CubeEnv)

start_time = time.time()
s_refs = [
    s.sample_once.remote(n_scramble=50, metric='half-turn')
    for _ in range(5000)
]
s_s = ray.get(s_refs)

print(f'Runtime: {time.time() - start_time:.3f} seconds, data:')
print(len(s_s))
