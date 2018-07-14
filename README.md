# TensorSpeak
TensorSpeak is an RNN written in TensorFlow for generating sentences. A model trained on 10k of tweets from the @realDonaldTrump twitter account can be found at [this S3 bucket](https://s3-us-west-2.amazonaws.com/noahs-ml-data/trump_2014-curr_cleaned_07-11--03-03.tar.gz)

For details about the model architecture and training visit [the blog post here](http://collaborativeai.org/projects/trump-tweet-generator/)

To generate tweets of your own using the model clone the repo or visit [the Google Colaboratory Notebook here](https://colab.research.google.com/drive/19HCUbO-NQZtqLsPtVHYj8Yu4xYbCJux7)

## Loading and Running the Model
To generate tweets from the model, download the trained model, unzip it, and copy the contents to the models directory.
Then you can run the generate.py script. Alternatively, run the train.py script to train the model. By default the script uses 
the trump_2014-curr_cleaned.csv in the data directory to train, but other text corpuses can be used. 

## Directories
**models** is where trained models are loaded from and saved to. 

**data** is where .csv files holding text corpuses are loaded from. 

**logs** is where TensorBoard logs are saved to. Start Tensorboard from within the main directory with "tensorboard --logdir=./logs/"

**preprocessors** is where Preprocessor objects are saved to and loaded from as pickles. These objects store parsed
training data. 

**src** contains all source python files. 
