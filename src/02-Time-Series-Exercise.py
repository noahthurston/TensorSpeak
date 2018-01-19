
# coding: utf-8

# # Time Series Exercise Example from Udemy for reference
# 
# ### Follow along with the instructions in bold. Watch the solutions video if you get stuck!

# ## The Data
# 
# ** Source: https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line **
# 
# **Monthly milk production: pounds per cow. Jan 62 - Dec 75**

# ** Import numpy pandas and matplotlib **

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

# ** Use pandas to read the csv of the monthly-milk-production.csv file and set index_col='Month' **

milk = pd.read_csv("../data/monthly-milk-production.csv", index_col='Month')


# ** Check out the head of the dataframe**
milk.head()



# ** Make the index a time series by using: **
# 
#     milk.index = pd.to_datetime(milk.index)

milk.index = pd.to_datetime(milk.index)

milk


# ** Plot out the time series data. **


milk.plot()
plt.show()


# ___

# ### Train Test Split
# 
# ** Let's attempt to predict a year's worth of data. (12 months or 12 steps into the future) **
# 
# ** Create a test train split using indexing (hint: use .head() or tail() or .iloc[]). We don't want a random train test split, we want to specify that the test set is the last 12 months of data is the test set, with everything before it is the training. **



len(milk)

training_set = milk.head(len(milk)-12)

testing_set = milk.tail(12)

# training_set

# testing_set



# ### Scale the Data
# 
# ** Use sklearn.preprocessing to scale the data using the MinMaxScaler. Remember to only fit_transform on the training data, then transform the test data. You shouldn't fit on the test data as well, otherwise you are assuming you would know about future behavior!**





scaler_model = MinMaxScaler()


training_set = scaler_model.fit_transform(training_set).astype(np.float32)

testing_set = scaler_model.transform(testing_set).astype(np.float32)

type(training_set[0,0])

training_set = training_set.astype(np.float32)

type(training_set[0,0])


# ## Batch Function
# 
# ** We'll need a function that can feed batches of the training data. We'll need to do several things that are listed out as steps in the comments of the function. Remember to reference the previous batch method from the lecture for hints. Try to fill out the function template below, this is a pretty hard step, so feel free to reference the solutions! **


def next_batch(training_data,batch_size,steps,start=-1):
    """
    INPUT: Data, Batch Size, Time Steps per batch
    OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
    """
    
    # STEP 1: Use np.random.randint to set a random starting point index for the batch.
    # Remember that each batch needs have the same number of steps in it.
    # This means you should limit the starting point to len(data)-steps
    
    if start == -1:
        rand_start = np.random.randint(0,len(training_data) - steps)
    else:
        rand_start = start
    # print("Starting at index %d out of %d with %d steps" % (rand_start,len(training_data), steps))
    
    # STEP 2: Now that you have a starting index you'll need to index the data from
    # the random start to random start + steps + 1. Then reshape this data to be (1,steps+1)
    
    full_set = np.array(training_data[rand_start:rand_start+steps+1].reshape(1,steps+1))
    
    # STEP 3: Return the batches. You'll have two batches to return y[:,:-1] and y[:,1:]
    # You'll need to reshape these into tensors for the RNN to .reshape(-1,steps,1)
    X_set = full_set[:,:-1].reshape(-1, steps, 1)
    y_set = full_set[:,1:].reshape(-1, steps, 1)

    # print(X_set)
    # print(y_set)
    
    return X_set, y_set




# ## Setting Up The RNN Model

# ** Import TensorFlow **




# ### The Constants
# 
# ** Define the constants in a single cell. You'll need the following (in parenthesis are the values I used in my solution, but you can play with some of these): **
# * Number of Inputs (1)
# * Number of Time Steps (12)
# * Number of Neurons per Layer (100)
# * Number of Outputs (1)
# * Learning Rate (0.03)
# * Number of Iterations for Training (4000)
# * Batch Size (1)


num_inputs = 1
num_timesteps = 12
num_neurons_inlayer = 100
num_outputs = 1
learning_rate = 0.03
num_iterations = 10000
batch_size = 1


# ** Create Placeholders for X and y. (You can change the variable names if you want). The shape for these placeholders should be [None,num_time_steps-1,num_inputs] and [None, num_time_steps-1, num_outputs] The reason we use num_time_steps-1 is because each of these will be one step shorter than the original time steps size, because we are training the RNN network to predict one point into the future based on the input sequence.**  

X_placeholder = tf.placeholder(tf.float32, [None, num_timesteps, num_inputs])
y_placeholder = tf.placeholder(tf.float32, [None, num_timesteps, num_outputs])


# ** Now create the RNN Layer, you have complete freedom over this, use tf.contrib.rnn and choose anything you want, OutputProjectionWrappers, BasicRNNCells, BasicLSTMCells, MultiRNNCell, GRUCell etc... Keep in mind not every combination will work well! (If in doubt, the solutions used an Outputprojection Wrapper around a basic LSTM cell with relu activation.**

cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons_inlayer, activation=tf.nn.relu),
                                              output_size=num_outputs)


# ** Now pass in the cells variable into tf.nn.dynamic_rnn, along with your first placeholder (X)**

outputs, states = tf.nn.dynamic_rnn(cell, X_placeholder, dtype=tf.float32)


# ### Loss Function and Optimizer
# 
# ** Create a Mean Squared Error Loss Function and use it to minimize an AdamOptimizer, remember to pass in your learning rate. **


loss = tf.reduce_mean(tf.square(outputs-y_placeholder)) #calc MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)


# ** Initialize the global variables **

init = tf.global_variables_initializer()


# ** Create an instance of tf.train.Saver() **
saver = tf.train.Saver()


# ### Session
# 
# ** Run a tf.Session that trains on the batches created by your next_batch function. Also add an a loss evaluation for every 100 training iterations. Remember to save your model after you are done training. **

with tf.Session() as sess:
    # CODE HERE!
    sess.run(init)
    
    print("starting")
    
    for iteration in range(num_iterations):
        
        #              def next_batch(training_data,batch_size,steps):
        X_batch, y_batch = next_batch(training_set, batch_size, num_timesteps)
        sess.run(train, feed_dict={X_placeholder: X_batch, y_placeholder: y_batch})
        
        if iteration % 100 == 0:
            
            mse = loss.eval(feed_dict={X_placeholder:X_batch, y_placeholder: y_batch})
            print(iteration, "\tMSE:", mse)
    
    # Save Model for Later
    saver.save(sess, "../data/ex_time_series_model_min")



# ______

# ## Predicting Future (Test Data)

# ** Show the test_set (the last 12 months of your original complete data set) **

# CODE HERE
plt.plot(training_set)


# ** Now we want to attempt to predict these 12 months of data, using only the training data we had. To do this we will feed in a seed training_instance of the last 12 months of the training_set of data to predict 12 months into the future. Then we will be able to compare our generated 12 months to our actual true historical values from the test set! **

# # Generative Session
# ### NOTE: Recall that our model is really only trained to predict 1 time step ahead, asking it to generate 12 steps is a big ask, and technically not what it was trained to do! Think of this more as generating new values based off some previous pattern, rather than trying to directly predict the future. You would need to go back to the original model and train the model to predict 12 time steps ahead to really get a higher accuracy on the test data. (Which has its limits due to the smaller size of our data set)
# ** Fill out the session code below to generate 12 months of data based off the last 12 months of data from the training set. The hardest part about this is adjusting the arrays with their shapes and sizes. Reference the lecture for hints.**

# def next_batch(training_data,batch_size,steps,start=-1):

with tf.Session() as sess:
    
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "../data/ex_time_series_model_min")

    training_seed = list(training_set[-12:])
    
    for iteration in range(12):

        X_batch = np.array(training_seed[-num_timesteps:]).reshape(1,num_timesteps,1)
        y_pred = sess.run(outputs, feed_dict={X_placeholder: X_batch})
        training_seed.append(y_pred[0, -1, 0])
        
        


# ** Show the result of the predictions. **

training_seed


# ** Grab the portion of the results that are the generated values and apply inverse_transform on them to turn them back into milk production value units (lbs per cow). Also reshape the results to be (12,1) so we can easily add them to the test_set dataframe.**


transformed_gs = scaler_model.inverse_transform(np.array(training_seed).reshape(-1,1))
transformed_gs


# ** Create a new column on the test_set called "Generated" and set it equal to the generated results. You may get a warning about this, feel free to ignore it.**


transformed_ts = scaler_model.inverse_transform(np.array(testing_set).reshape(-1,1))
transformed_ts


# ** View the test_set dataframe. **
# ** Plot out the two columns for comparison. **



plt.plot(range(1,13), transformed_gs[-12:], "b-")
plt.plot(range(1,13), transformed_ts[-12:], "r-")
plt.show()

transformed_ts_all = scaler_model.inverse_transform(np.array(training_set).reshape(-1,1))

len(transformed_ts_all)

plt.plot(range(144,156), transformed_gs[-12:], "b-")
plt.plot(range(1,157), transformed_ts_all[:], "r-")
plt.show()

# # Great Job!
# 
# Play around with the parameters and RNN layers, does a faster learning rate with more steps improve the model? What about GRU or BasicRNN units? What if you train the original model to not just predict one timestep ahead into the future, but 3 instead? Lots of stuff to add on here!










