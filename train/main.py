import ray

import time

ray.init()

database = [
    'Learning', 'Ray', 'Flexible', 'Distributed', 'Python', 'for', 'Data',
    'Science'
]

database_ref = ray.put(database)


@ray.remote
def retrieve_task(item):
    obj_store_data = ray.get(database_ref)
    time.sleep(item / 10.)
    return item, obj_store_data[item]


def retrieve(item):
    time.sleep(item / 10.)
    return item, database[item]


def print_runtime(input_data, start_time, decimals=1):
    print(f'Runtime: {time.time() - start_time:.{decimals}f} seconds, data:')
    print(*input_data, sep="\n")


@ray.remote
class DataTracker:
    def __init__(self, ):
        self._counts = 0

    def increment(self):
        self._counts += 1

    def counts(self):
        return self._counts


@ray.remote
def retrieve_tracker_task(item, tracker: DataTracker):
    obj_store_data = ray.get(database_ref)
    time.sleep(item / 10.)
    tracker.increment.remote()
    return item, obj_store_data[item]


tracker = DataTracker.remote()

data_references = [
    retrieve_tracker_task.remote(item, tracker) for item in range(8)
]
data = ray.get(data_references)
print(ray.get(tracker.counts.remote()))

start = time.time()
data_ref = [retrieve_task.remote(item) for item in range(8)]
data = ray.get(data_ref)

print_runtime(data, start, 2)

start = time.time()
data_refs = [retrieve_task.remote(item) for item in range(8)]
all_data = []

while len(data_refs) > 0:
    finished, data_refs = ray.wait(data_refs, num_returns=2, timeout=0.7)
    data = ray.get(finished)
    print_runtime(data, start, 3)
    all_data.extend(data)


@ray.remote
def follow_up_task(retrieve_result):
    original_item, _ = retrieve_result
    follow_up_result = retrieve(original_item + 1)
    return retrieve_result, follow_up_result


retrieved_refs = [retrieve_task.remote(item) for item in [0, 2, 4, 6]]
follow_up_refs = [follow_up_task.remote(ref) for ref in retrieved_refs]

result = [print(data) for data in ray.get(follow_up_refs)]
