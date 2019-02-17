# tf_cpp

This toy project demonstrates how to:
  * Use TensorFlow's C++ API to generate training and test data sets, in TFRecord format;
  * Load the data sets in Python with TFRecordDataset, train a model with Keras and export the model to a H5 file.
  * Convert the model in .H5 to Tensorflow's GraphDef.
  * Load the model and run inference with TensorFlow's C++ API.
  
The demo is based on the works in:
  * [minigo](https://github.com/tensorflow/minigo): for configuring TensorFlow in Bazel. ./cc/configure.sh is copied from [the project](https://github.com/tensorflow/minigo/blob/master/cc/configure_tensorflow.sh) with minor modifications;
  * [keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow): for converting a Keras model to a TensorFlow model. ./keras_to_tensorflow.py is copied from [the project](https://github.com/amir-abdi/keras_to_tensorflow/blob/master/keras_to_tensorflow.py) with minor modifications.
  * [MNIST](http://yann.lecun.com/exdb/mnist/): the training and test datasets are generated from the MNIST database.

## Run the Demo

To run the demo, you need to install the following:
* Python
* TensorFlow
* Bazel

### Configure and Build Tensorflow
```bash
chmod +x cc/configure_tf.sh \
./cc/configure_tf.sh
```

### Generate TFRecords from C++
Download [the MNIST dataset](http://yann.lecun.com/exdb/mnist/), extract the files to a work directory, e.g. `/tmp/mnist`.
```bash
work_dir=/tmp/mnist
mkdir -p ${work_dir}

bazel build -c opt cc:gen_dataset

bazel-bin/cc/gen_dataset \
  --images=${work_dir}/train-images-idx3-ubyte \
  --labels=${work_dir}/train-labels-idx1-ubyte \
  --records=${work_dir}/mnist_train.rio

bazel-bin/cc/gen_dataset \
  --images=${work_dir}/t10k-images-idx3-ubyte \
  --labels=${work_dir}/t10k-labels-idx1-ubyte \
  --records=${work_dir}/mnist_test.rio
```

### Train with Keras
```bash
python mnist_nn.py \
  --train_records=${work_dir}/mnist_train.rio \
  --test_records=${work_dir}/mnist_test.rio \
  --use_cnn={true|false} \
  --trained_model=${work_dir}/my_model.h5 \
  --first_layer_name=mnist
```
Choose a unique value for the flag `--first_layer_name`. The input node's name will be "${first_layer_name}_input" and run_model.cc needs to know this name.

### Convert the Keras Model to TensorFlow Format
```bash
python keras_to_tensorflow.py \
  --input_model=${work_dir}/my_model.h5 \
  --output_model=${work_dir}/my_model.pb \
  --output_nodes_prefix=mnist_output/
```
If the model has only one output, the final output node's name will be "${output_nodes_prefix}0$" and run_model.cc will use this name.

### Run the Model in C++
```bash
bazel build -c opt cc:run_model

bazel-bin/cc/run_model \
  --model=./tmp/my_model.pb \
  --input_layer_name=mnist_input \
  --output_layer_name=mnist_output/0 \
  --v=1
```

If the program complains about any compatibility issue, try to re-install tensorflow with the package built from `cc/configure_tf.sh`.
```bash
pip uninstall ${your-tensorflow-version}
pip install ${tmp_pkg_dir}/tensorflow-{your-version-tags}.whl
```
