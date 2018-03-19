import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

from collections import namedtuple

import timeit







big_arr = np.array([[1,2], [3,4]])
small_arr = np.array([[5,6]])

sum = np.append(big_arr, small_arr).reshape(-1, 2)

print(sum)















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


"""
    def graph_mse(self):
        x_values = np.array(range(len(self.historical_mse)))

        title_str = ("%s, TS=%d, NeurPL=%d, LR=%.4f, Iters=%d" % (self.corpus_file_name, self.num_timesteps, self.num_neurons_inlayer, self.learning_rate, self.num_iterations))
        plt.plot(x_values, self.historical_mse)
        plt.title(title_str)

        t = time.asctime(time.localtime(time.time()))
        save_str = self.save_dir + "graph_" + self.corpus_file_name + "_" + t + "_.png"

        plt.savefig(save_str, format='png', dpi=300)
"""
