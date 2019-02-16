load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# Google libs
#
http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-master",
    urls = ["https://github.com/abseil/abseil-cpp/archive/master.zip"],
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "94ad0467a0de3331de86216cbc05636051be274bf2160f6e86f07345213ba45b",
    strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
    url = "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.zip",
)

http_archive(
    name = "com_github_google_glog",
    urls = ["https://github.com/google/glog/archive/55cc27b6eca3d7906fc1a920ca95df7717deb4e7.tar.gz"],
    sha256 = "4966f4233e4dcac53c7c7dd26054d17c045447e183bf9df7081575d2b888b95b",
    strip_prefix = "glog-55cc27b6eca3d7906fc1a920ca95df7717deb4e7",
)
