cc_binary(
    name = "gen_dataset",
    srcs = ["gen_dataset.cc"],
    deps = [
      ":tensorflow_dynamic",
      "@com_google_absl//absl/strings",
      "@com_github_gflags_gflags//:gflags",
      "@com_github_google_glog//:glog",
    ],
)

cc_binary(
    name = "run_model",
    srcs = ["run_model.cc"],
    deps = [
      ":tensorflow_dynamic",
      "@com_google_absl//absl/algorithm",
      "@com_google_absl//absl/container:inlined_vector",
      "@com_google_absl//absl/memory",
      "@com_google_absl//absl/strings",
      "@com_google_absl//absl/types:span",
      "@com_github_gflags_gflags//:gflags",
      "@com_github_google_glog//:glog",
    ],
)

cc_library(
    name = "tensorflow_dynamic",
    srcs = [
        "tensorflow/libtensorflow_framework.so",
        "tensorflow/libtensorflow_cc.so",
    ],
    hdrs = glob(
        ["tensorflow/**/*"],
        exclude = [
            "**/*.a",
            "**/*.so",
            "tensorflow/flatbuffers/**",
            "tensorflow/tensorflow/contrib/lite/**",
        ],
    ),
    includes = [
        "tensorflow",
    ],
    tags = ["manual"],
)
