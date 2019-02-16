# tf_cpp

This toy project demonstrates:
  * Use TensorFlow's C++ API to generate training and test data sets, in TFRecord format;
  * Load the data sets in Python with TFRecordDataset, train a model with Keras and export the model to a H5 file.
  * Convert the model in .H5 to Tensorflow's GraphDef.
  * Load the model and run inference with TensorFlow's C++ API.
  
The demo is based on the works in:
  * [minigo](https://github.com/tensorflow/minigo/blob/master/README.md): for configuring TensorFlow in Bazel. ./cc/configure.sh is copied from [the project](https://github.com/tensorflow/minigo/blob/master/cc/configure_tensorflow.sh) with minor modifications;
  * [keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow/blob/master/README.md): for converting a Keras model to a TensorFlow model. ./keras_to_tensorflow.py is copied from [the project](https://github.com/amir-abdi/keras_to_tensorflow/blob/master/keras_to_tensorflow.py) with minor modifications.
  * [MNIST](http://yann.lecun.com/exdb/mnist/): training and test datasets are generated from the MNIST database.

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
  --trained_model=${work_dir}/my_model.h5 \
  --input_layer_name=mnist
```

### Convert Keras Model to TensorFlow Format
```bash
python keras_to_tensorflow.py \
  --input_model=${work_dir}/my_model.h5 \
  --output_model=${work_dir}/my_model.pb \
  --output_nodes_prefix=mnist_output/
```

### Run the Model in C++
```bash
bazel build -c opt cc:run_model

bazel-bin/cc/run_model \
  --model=./tmp/my_model.pb \
  --input_layer_name=mnist_input \
  --output_layer_name=mnist_output/0 \
  --v=1
```
