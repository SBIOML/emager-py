import emager_py.data.emager_redis as er
import numpy as np
import emager_py.finn.remote_operations as ro

print("*" * 100)
print(
    "redis_remote_operations.py: This script requires a Redis server to be running. To easily do so, `docker run --name emager-redis redis`."
)
print("*" * 100)

r = er.EmagerRedis("localhost")
r.clear_data()
r.set_sampling_params(100, 50, 500)
r.set_rhd_sampler_params(
    20, 300, 0, 15, ro.DEFAULT_EMAGER_PYNQ_PATH + "/bitfile/finn-accel.bit"
)
r.push_sample(np.random.randint(0, 100, (64, 64)), np.random.randint(0, 6, 64))
print(len(r.pop_sample(False, 1)))
print(len(r.pop_sample(True, 1)))
