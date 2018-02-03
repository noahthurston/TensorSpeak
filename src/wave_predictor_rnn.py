### rnn to predict the next y coord of a simple wave

import numpy as np
import matplotlib.pyplot as plt
import sys



from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

MyModelDesc = {"num_io":4,
           "num_timesteps":6,
           "num_neurons_inlayer":100,
           "learning_rate":0.030,
           "num_iterations":1000,
           "batch_size":1}

# create placeholders for current timestep data (X_placeholder) and next timestep data (y_placeholder)
X_placeholder = tf.placeholder(tf.float32, [None, MyModelDesc["num_timesteps"], MyModelDesc["num_io"]])
y_placeholder = tf.placeholder(tf.float32, [None, MyModelDesc["num_timesteps"], MyModelDesc["num_io"]])

# create RNN layer, in this case a LSTM with relu activation
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=MyModelDesc["num_neurons_inlayer"], activation=tf.nn.relu),output_size=MyModelDesc["num_io"])

# ** Now pass in the cells variable into tf.nn.dynamic_rnn, along with your first placeholder (X)**

# outputs are what is calculated by the cell, state is the current LSTM state
outputs, states = tf.nn.dynamic_rnn(cell, X_placeholder, dtype=tf.float32)

# use a softmax with cross entropy since the outputs are a vector of probabilities that sum to 1
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=y_placeholder))

# create optimizer tha trains at the given learning_rate
optimizer = tf.train.AdamOptimizer(learning_rate=MyModelDesc["learning_rate"])
train = optimizer.minimize(loss)

# initialize globals
init = tf.global_variables_initializer()

# create saver to save model later
saver = tf.train.Saver()

# ### Session
#
# ** Run a tf.Session that trains on the batches created by your next_batch function. Also add an a loss evaluation for every 100 training iterations. Remember to save your model after you are done training. **


#    X_set = full_set[:,:-1].reshape(-1, steps, 1)
#    y_set = full_set[:,1:].reshape(-1, steps, 1)


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
    # print("random_start: %d" % random_start)

    X_batch = test_arr_x5[:, random_start:random_start + ModelDesc["num_timesteps"], :]
    y_batch = test_arr_x5[:, random_start + 1:random_start + ModelDesc["num_timesteps"] + 1, :]

    return X_batch, y_batch






############## BEGIN TRAINING ##########


with tf.Session() as sess:
    sess.run(init)

    # list of MSEs for graphing later
    mse_list = np.zeros(int(MyModelDesc["num_iterations"] / 100) + 1)
    print(mse_list)

    for iteration in range(MyModelDesc["num_iterations"]):
        iter_div = int(iteration / 100)

        X_batch, y_batch = get_next_batch(MyModelDesc)
        print("X_batch:")
        print(X_batch)
        print("y_batch:")
        print(y_batch)

        # X_batch, y_batch = next_batch(training_set, batch_size, num_timesteps)
        sess.run(train, feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})

        # every 100 iterations, print progress
        if iteration % 100 == 0:

            mse_list[iter_div] = loss.eval(feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})

            if mse_list[iter_div - 1] < mse_list[iter_div]:
                print("%f < %f" % (mse_list[iter_div - 1], mse_list[iter_div]))
                learning_rate = MyModelDesc["learning_rate"] / 2
                print("Halving learning rate: %f to %f" % (MyModelDesc["learning_rate"] * 2, MyModelDesc["learning_rate"]))

            print(iteration, "\tMSE:", mse_list[iter_div])
            # print(mse_list)

    # Save Model for Later
    saver.save(sess, "../data/wave_predictor_rnn")

x_plt = np.arange(0, int(MyModelDesc["num_iterations"] / 100) + 1, 1).reshape(-1, 1)
y_plt = mse_list.reshape(-1, 1)
print(x_plt)
print(y_plt)

plt.show()
plt.plot(x_plt, y_plt, "r-")
plt.show()






############## BEGIN MAKING PREDICTIONS ##########


with tf.Session() as sess:
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "../data/wave_predictor_rnn")

    X_seed, y_true = get_next_batch(MyModelDesc)
    y_pred = sess.run(tf.nn.softmax(logits=outputs), feed_dict={X_placeholder: X_seed})
    print("\n")
    print(X_seed)

    print(y_true)
    print(y_pred)

    y_true_translated = []
    y_pred_translated = []

    for row in y_true[0]:
        y_true_translated.append(row[:].argmax())

    for row in y_pred[0]:
        y_pred_translated.append(row[:].argmax())

    print(y_true_translated)
    print(y_pred_translated)


    plt.plot(range(0,6), y_true_translated,"g-")
    plt.plot(range(0, 6), y_pred_translated, "ro")
    plt.show()

    print(np.array(X_seed[0, 0, :]).argmax())
    print(np.array(X_seed[0, 1, :]).argmax())
    print(np.array(X_seed[0, 2, :]).argmax())
    print(np.array(X_seed[0, 3, :]).argmax())
    print(np.array(X_seed[0, 4, :]).argmax())
    print(np.array(X_seed[0, 5, :]).argmax())

sys.exit()



