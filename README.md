# tf_cpp

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

### Train
```bash
python mnist_nn.py \
  --train_records=${work_dir}/mnist_train.rio \
  --test_records=${work_dir}/mnist_test.rio \
  --trained_model=${work_dir}/my_model.h5 \
  --input_layer_name=mnist
```

### Convert
```bash
python keras_to_tensorflow.py \
  --input_model=${work_dir}/my_model.h5 \
  --output_model=${work_dir}/my_model.pb \
  --output_nodes_prefix=mnist_output/
```

### Run
```bash
bazel build -c opt cc:run_model

bazel-bin/cc/run_model \
  --model=./tmp/my_model.pb \
  --input_layer_name=mnist_input \
  --output_layer_name=mnist_output/0 \
  --v=1
```
