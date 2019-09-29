# sentiment analysis using tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import graphical lib
import matplotlib.pyplot as plt
import numpy as np

# import tensorflow framework for
import neural_structured_learning as nsl
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

import tensorflow_hub as hub
tf.keras.backend.clear_session()

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
# print("GPU is ", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# load imdb movie reviews database
imdb_dataset = tf.keras.datasets.imdb

# fetch train and test data, num_words=10000 indicates that you should keep 10000 most frequently occuring words in the training data
(train_data, train_labels), (test_data, test_labels) = (imdb_dataset.load_data(num_words=10000))

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

train_data_size = len(train_data)

def build_reverse_word_index():
    imdb_word_index = imdb_dataset.get_word_index()
    imdb_word_index = {k: (v + 3) for k, v in imdb_word_index.items()}
    imdb_word_index['<PAD>'] = 0
    imdb_word_index['<START>'] = 1
    imdb_word_index['<UNK>'] = 2
    imdb_word_index['<UNUSED>'] = 3
    return dict((value, key) for (key, value) in imdb_word_index.items())

imdb_reverse_word_index = build_reverse_word_index()

def decode_review(review):
    return ' '.join([imdb_reverse_word_index.get(sample_key, '?') for sample_key in review])

print("Example of vector translation using word index from {}".format(train_data[0]))
print("to {}".format(decode_review(train_data[0])))
