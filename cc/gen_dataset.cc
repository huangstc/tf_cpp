#include <cstdio>
#include <string>

#include "absl/strings/str_cat.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

DEFINE_string(images, "", "MNIST image file.");
DEFINE_string(labels, "", "MNIST label file.");
DEFINE_string(records, "", "Output TFRecords to this file.");

// mnist_nn.py will need these names to parse examples.
static const char kImageFeatureName[] = "pix";
static const char kLabelFeatureName[] = "label";

using tensorflow::io::RecordWriter;
using tensorflow::io::RecordWriterOptions;
using tensorflow::Example;

struct Image {
  int32_t width;
  int32_t height;
  int8_t label;
  std::vector<float> pixels;
};

uint32_t ToInt32(const char* buf) {
  uint32_t num = 0;
  for (int i = 0; i < 4; ++i) {
    num *= 256;
    num += static_cast<uint8_t>(buf[i]);
  }
  return num;
}

char* ReadFileToBuffer(const std::string& path) {
  FILE* file = nullptr;
  file = fopen(path.data(), "rb");
  CHECK(file != nullptr) << "Failed in opening file " << path;

  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  rewind(file);

  char* buffer = new char[file_size];
  CHECK(buffer != nullptr) << "Memory error for " << path;

  size_t result = fread(buffer, 1, file_size, file);
  CHECK_EQ(file_size, result) << "Reading error for " << path;

  fclose(file);
  return buffer;
}

std::vector<Image> LoadMnist(const std::string& image_file,
                             const std::string& label_file) {
  char* image_buffer = ReadFileToBuffer(image_file);
  char* label_buffer = ReadFileToBuffer(label_file);

  char* header = image_buffer;
  const uint32_t num_images = ToInt32(header + 4);
  const int32_t height = ToInt32(header + 8);
  const int32_t width = ToInt32(header + 12);
  LOG(INFO) << "Num images: " << num_images << ", width=" << width
            << ", height=" << height;

  header = label_buffer;
  const uint32_t num_labels = ToInt32(header + 4);
  CHECK_EQ(num_images, num_labels);
  LOG(INFO) << "Num labels: " << num_labels;

  std::vector<Image> images;
  images.resize(num_images);
  char* cur = image_buffer + 16;
  for (size_t i = 0; i < num_images; ++i) {
    images[i].width = width;
    images[i].height = height;
    images[i].pixels.reserve(width * height);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        uint8_t value = static_cast<uint8_t>(*cur++);
        images[i].pixels.push_back(static_cast<float>(value) / 255.0);
      }
    }
  }

  char* buffer = label_buffer + 8;
  for (size_t i = 0; i < num_images; ++i) {
    images[i].label = static_cast<int8_t>(buffer[i]);
  }

  delete image_buffer;
  delete label_buffer;
  return images;
}

tensorflow::Feature MakeFloatsFeature(const std::vector<float>& ff) {
  tensorflow::Feature feature;
  auto* float_list = feature.mutable_float_list();
  for (const float f : ff) {
    float_list->add_value(f);
  }
  return feature;
}

Example ImageToExample(const Image& image) {
  Example example;
  auto& features = *example.mutable_features()->mutable_feature();
  features[kImageFeatureName] = MakeFloatsFeature(image.pixels);
  features[kLabelFeatureName].mutable_int64_list()->add_value(image.label);
  return example;
}

std::vector<Example> ImagesToExamples(const std::vector<Image>& images) {
  std::vector<Example> examples;
  examples.reserve(images.size());
  for (const auto& i : images) {
    examples.push_back(ImageToExample(i));
  }
  return examples;
}

void WriteExamples(const std::string& path,
                   const std::vector<Example>& examples) {
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(path, &file));

  RecordWriterOptions options;
  options.compression_type = RecordWriterOptions::ZLIB_COMPRESSION;
  RecordWriter writer(file.get(), options);

  std::string data;
  for (const auto& example : examples) {
    example.SerializeToString(&data);
    TF_CHECK_OK(writer.WriteRecord(data));
  }
  TF_CHECK_OK(writer.Close());
  TF_CHECK_OK(file->Close());
}

void Convert(const std::string& image_file, const std::string& label_file,
             const std::string& tfrecord_output) {
  auto images = LoadMnist(image_file, label_file);
  auto examples = ImagesToExamples(images);
  LOG(INFO) << "First example: " << examples[0].DebugString();
  WriteExamples(tfrecord_output, examples);
  LOG(INFO) << "TFRecords are written to " << tfrecord_output;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  Convert(FLAGS_images, FLAGS_labels, FLAGS_records);
  return 0;
}
