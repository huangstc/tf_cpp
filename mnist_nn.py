import numpy as np
import tensorflow as tf

from absl import app, flags
from tensorflow import keras

flags.DEFINE_string('train_records', None, 'Path to the training dataset.')
flags.DEFINE_string('test_records', None, 'Path to the test dataset.')
flags.DEFINE_string('trained_model', None, 'Path to the saved model, ending with .h5')
flags.DEFINE_string('input_layer_name', 'mnist', 'Name the input layer.')

FLAGS = flags.FLAGS

# Parameters
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_FEATURE_NAME = 'pix'          # Also defined in cc/gen_dataset.cc
LABEL_FEATURE_NAME = 'label'        # Also defined in cc/gen_dataset.cc
SHUFFLE_BUFFER = 10000
BATCH_SIZE = 60
STEPS_PER_TRAIN_EPOCH = 1000
NUM_CLASSES = 10

# Transforms example_proto into a pair of a scalar integer and a float 2d array,
# representing an image and its label, respectively.
def _parse_function(example_proto):
    features = {LABEL_FEATURE_NAME: tf.FixedLenFeature([], tf.int64),
                IMAGE_FEATURE_NAME: tf.VarLenFeature(tf.float32)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.reshape(tf.sparse.to_dense(parsed_features[IMAGE_FEATURE_NAME]),
                       [IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    return image, parsed_features[LABEL_FEATURE_NAME]


def create_dataset(files):
    # Also set in cc/gen_dataset.cc
    dataset = tf.data.TFRecordDataset(files, "ZLIB")
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def main(argv):
    del argv  # Unused
    flags.mark_flag_as_required('train_records')
    flags.mark_flag_as_required('test_records')
    flags.mark_flag_as_required('trained_model')

    train_dataset = create_dataset([FLAGS.train_records])
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER)
    test_dataset = create_dataset([FLAGS.test_records])

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1),
                             name=FLAGS.input_layer_name),
        keras.layers.Dense(100, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print(model.summary())

    model.fit(train_dataset, epochs=10, steps_per_epoch=STEPS_PER_TRAIN_EPOCH)

    loss, acc = model.evaluate(test_dataset, steps=94)
    print("test loss: %f" % loss)
    print("test accuracy: %f" % acc)

    print("The input layer name is %s" % FLAGS.input_layer_name)
    print("Model will be save to %s" % FLAGS.trained_model)
    model.save(FLAGS.trained_model)


if __name__ == '__main__':
    app.run(main)
