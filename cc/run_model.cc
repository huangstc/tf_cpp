#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

DEFINE_string(model, "", "Load model from this file.");
DEFINE_string(input_layer_name, "mnist_input", "");
DEFINE_string(output_layer_name, "mnist_output/0", "");

// Parameters:
static const int kImageWidth = 28;
static const int kImageHeight = 28;

namespace tf = tensorflow;

using std::string;

// Reads a model from the output of keras_to_tensorflow.py
// Returns a Session object that loads the model graph.
std::unique_ptr<tf::Session> LoadGraph(const std::string& graph_file_name) {
  tf::GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(tf::Env::Default(), graph_file_name, &graph_def));
  for (const auto& node : graph_def.node()) {
    LOG(INFO) << "Node:" << node.name() << " op:" << node.op();
  }
  VLOG(1) << graph_def.DebugString();
  auto sess = absl::WrapUnique<tf::Session>(
      tf::NewSession(tf::SessionOptions()));
  TF_CHECK_OK(sess->Create(graph_def));
  return sess;
}

// Returns a Tensor representing a dummy gray image.
tf::Tensor ReadInput() {
  tf::Tensor example(
      tf::DT_FLOAT,
      tf::TensorShape({/*batch size*/1, kImageWidth, kImageWidth,
                       /*channels*/1}));
  auto tensor = example.tensor<float, /*dim*/4>();
  for (int i = 0; i < kImageHeight; ++i) {
    for (int j = 0; j < kImageWidth; ++j) {
      tensor(0, i, j, 0) = 0.5;
    }
  }
  return example;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  auto sess = LoadGraph(FLAGS_model);
  tf::Tensor input = ReadInput();
  std::vector<tf::Tensor> outputs;
  TF_CHECK_OK(sess->Run({{FLAGS_input_layer_name, input}},
                         {FLAGS_output_layer_name}, {}, &outputs));

  tf::TensorProto proto;
  outputs[0].AsProtoField(&proto);
  LOG(INFO) << proto.DebugString();

  return 0;
}
