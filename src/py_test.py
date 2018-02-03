import numpy as np
from collections import namedtuple

import timeit


vocab_size = 10
target_index = 3
time = timeit.timeit('innef_vectorized_sent = [1 if x == 3 else 0 for x in range(8000)]', number=10)
print(time)
#print(innef_vectorized_sent)

print("\n\n")

#timeit.timeit('vectorized_sent = np.zeros(10)')
#vectorized_sent = vectorized_sent.tolist()
#vectorized_sent[target_index] = 1
#print(vectorized_sent)

"""




## num inputs and outputs = 5, which would be the same as the vocab size
num_inputs = 4
num_outputs = 4

num_timesteps = 6
num_neurons_inlayer = 100
learning_rate = 0.030
num_iterations = 1000
batch_size = 1
"""


"""
lets create some static test data to test the tensor inputs
goal:
[[[1,0,0,0]
  [0,1,0,0]
  [0,0,1,0]
  [0,0,0,1]
  [0,0,1,0]
  [0,1,0,0]
  [1,0,0,0]]]

















MyModelDesc = {"num_io":4,
           "num_timesteps":6,
           "num_neurons_inlayer":100,
           "learning_rate":0.030,
           "num_iterations":1000,
           "batch_size":1}








def get_next_batch(ModelDesc):
    test_arr = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    test_arr_x5 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

    num_test_vectors = len(test_arr_x5) / ModelDesc["num_io"]

    # print("len(test_arr_x5): %d" % len(test_arr_x5))
    # print("num_test_vectors: %d" % num_test_vectors)

    test_arr_x5 = np.array(test_arr_x5).reshape(1, -1, ModelDesc["num_io"])

    # print("len(test_arr_x5): %d" % len(test_arr_x5))
    # print("num_test_vectors: %d" % num_test_vectors)

    # print(test_arr_x5)

    # we feed it 4 inputs, so we can ask for anything a starting point anywhere between 0 -> -5
    # last possible X_batch: -5 -> -2
    # last possible y_batch: -4 -> -1

    smallest_rand_start = 0
    largest_rand_start = int(num_test_vectors - ModelDesc["num_timesteps"] - 1)

    random_start = np.random.randint(smallest_rand_start, largest_rand_start)
    print("random_start: %d" % random_start)

    X_batch = test_arr_x5[:, random_start:random_start + ModelDesc["num_timesteps"], :]
    y_batch = test_arr_x5[:, random_start + 1:random_start + ModelDesc["num_timesteps"] + 1, :]

    return X_batch, y_batch


for iteration in range(10):
    print("iteration: %d" % iteration)
    my_X_batch, my_y_batch = get_next_batch(MyModelDesc)

    print(my_X_batch)
    print(my_y_batch)

    print("\n\n")

#    X_set = full_set[:,:-1].reshape(-1, steps, 1)
#    y_set = full_set[:,1:].reshape(-1, steps, 1)
"""