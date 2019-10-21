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

# it is time for a graph construction
# creating embeddings for text samples
# we use pretrained Swivel ebmeddings
# resulting embeddings will be store in a TFRecord format with additional feature that represents the ID of each sample

# This is necessary because hub.KerasLayer assumes tensor hashability, which
# is not supported in eager mode.
tf.compat.v1.disable_tensor_equality()

pretrained_embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'

hub_layer = hub.KerasLayer(pretrained_embedding, input_shape=[], dtype=tf.string, trainable=True)

def _int64_feature(value):
    # Returns int64 tf.train.Feature
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))

def _bytes_feature(value):
    # Returns bytes tf.train.Feature
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _float_feature(value):
    # Returns float tf.train.Feature
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))

def create_embedding_example(word_vector, record_id):
    # Create tf.Example containing sample's embedding and ID
    text = decode_review(word_vector)

    # Shape = [batch_size,]
    sentence_embedding = hub_layer(tf.reshape(text, shape=[-1,]))

    # Flatten the sentence embedding to 1-D
    sentence_embedding = tf.reshape(sentence_embedding, shape=[-1])

    features = {
        'id': _bytes_feature(str(record_id)),
        'embedding': _float_feature(sentence_embedding.numpy())
    }

    return tf.train.Example(features=tf.train.Features(feature=features))

def create_embeddings(word_vectors, output_path, starting_record_id):
    record_id = int(starting_record_id)
    with tf.io.TFRecordWriter(output_path) as writer:
        for word_vector in word_vectors:
            example = create_embedding_example(word_vector, record_id)
            record_id = record_id + 1
            writer.write(example.SerializeToString())
        return record_id

#create_embeddings(train_data, 'tmp/imdb/embeddings.tfr', 0)

def create_example(word_vector, label, record_id):
    features = {
        'id': _bytes_feature(str(record_id)),
        'words': _int64_feature(np.asarray(word_vector)),
        'label': _int64_feature(np.asarray([label]))
    }

    return tf.train.Example(features=tf.train.Features(feature=features))

def create_records(word_vectors, labels, record_path, starting_record_id):
    record_id = int(starting_record_id)
    with tf.io.TFRecordWriter(record_path) as writer:
        for word_vector, label in zip(word_vectors, labels):
            example = create_example(word_vector, label, record_id)
            record_id = record_id + 1
            writer.write(example.SerializeToString())
        return record_id


#next_record_id = create_records(train_data, train_labels, 'tmp/imdb/train_data.tfr', 0)
#create_records(test_data, test_labels, 'tmp/imdb/test_data.tfr', next_record_id)

NBR_FEATURE_PREFIX = "NL_nbr_"
NBR_WEIGHT_SUFIX = "_weight"

class HParams:
    def __init__(self):
        ### dataset parameters
        self.num_classes = 2
        self.max_seq_length = 256
        self.vocab_size = 10000
        ### neural graph learning parameters
        self.distance_type = nsl.configs.DistanceType.L2
        self.graph_regularization_multiplier = 0.1
        self.num_neighbors = 2
        ### model architecture
        self.num_embedding_dims = 16
        self.num_lstm_dims = 64
        self.num_fc_units = 64
        ### training parameters
        self.train_epochos = 10
        self.batch_size = 128
        ### eval parameters
        self.eval_steps = None

HPARAMS = HParams()

def pad_sequence(sequence, max_seq_length):
    pad_size = tf.maximum([0], max_seq_length - tf.shape(sequence)[0])
    padded = tf.concat([sequence.values, tf.fill((pad_size), tf.cast(0, sequence.dtype))], axis=0)
    return tf.slice(padded, [0], [max_seq_length])


def parse_example(example_proto):
    print("example_proto")
    print(example_proto)
    feature_spec = {
        'words': tf.io.VarLenFeature(tf.int64),
        'label': tf.io.FixedLenFeature((), tf.int64, default_value=-1)
    }
    print("feature_spec")
    print(feature_spec)

    for i in range(HPARAMS.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'words')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i, NBR_WEIGHT_SUFIX)
        feature_spec[nbr_feature_key] = tf.io.VarLenFeature(tf.int64)
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature([1], tf.float64, default_value =tf.constant([0.0]))

    features = tf.io.parse_single_example(example_proto, feature_spec)
    features['words'] = pad_sequence(features['words'], HPARAMS.max_seq_length)

    print("features")
    print(features)

    for i in range(HPARAMS.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'words')
        features[nbr_feature_key] = pad_sequence(features[nbr_feature_key], HPARAMS.max_seq_length)

    print("featuresv2")
    print(features)

    labels = features.pop('label')
    return features, labels

def make_dataset(file_path, training=False):
    dataset = tf.data.TFRecordDataset([file_path])
    if training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.map(parse_example)
    dataset = dataset.batch(HPARAMS.batch_size)
    return dataset

train_dataset = make_dataset('tmp/imdb/nsl_train_data.tfr', True)
test_dataset = make_dataset('tmp/imdb/test_data.tfr')

print(train_dataset)
print(test_dataset)

def make_feed_forward_model():
    inputs = tf.keras.Input(shape=(HPARAMS.max_seq_length,), dtype='int64', name='words')
    embedding_layer = tf.keras.layers.Embedding(HPARAMS.vocab_size, 16)(inputs)
    pooling_layer = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)
    dense_layer = tf.keras.layers.Dense(16, activation='relu')(pooling_layer)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def make_bilstm_model():
    inputs = tf.keras.Input(shape=(HPARAMS.max_seq_length,), dtype='int64', name='words')
    embedding_layer = tf.keras.layers.Embedding(HPARAMS.vocab_size, HPARAMS.num_embedding_dims)(inputs)

    lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HPARAMS.num_lstm_dims))(embedding_layer)

    dense_layer = tf.keras.layers.Dense(HPARAMS.num_fc_units, activation='relu')(lstm_layer)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

create_new_model = True

if create_new_model:
    bilstm_model = make_bilstm_model()
    bilstm_model.summary()

    feed_forward_model = make_feed_forward_model()
    feed_forward_model.summary()

    bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    validation_fraction = 0.9
    validation_size = int(validation_fraction * int(train_data_size / HPARAMS.batch_size))

    print("Validation set size {}".format(validation_size))
    validation_dataset = train_dataset.take(validation_size)
    train_dataset = train_dataset.skip(validation_size)

    history = bilstm_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=HPARAMS.train_epochos,
        verbose=1
    )

    results = bilstm_model.evaluate(test_dataset, steps=HPARAMS.eval_steps)
    print(results)

    history_dict = history.history
    print(history_dict.keys())

    bilstm_model.save('./models/bilstm_keras_model.h5')

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)

    #plt.plot(epochs, loss, '-r^', label='Training label')
    #plt.plot(epochs, val_loss, '-bo', label='Validation label')
    #plt.title('Training and validation loss')
    plt.plot(epochs, acc, '-r^', label='Training label')
    plt.plot(epochs, val_acc, '-bo', label='Validation label')
    plt.title('Training and validation accuracy')
    plt.xlabel('epochs')
    #plt.ylabel('loss')
    plt.ylabel('accuracy')
    plt.legend(loc='best')

    plt.show()
else:
    bilstm_model = tf.keras.models.load_model('./models/bilstm_keras_model.h5')