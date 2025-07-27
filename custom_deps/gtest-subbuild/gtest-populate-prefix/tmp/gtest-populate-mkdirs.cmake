# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/root/bmf/custom_deps/gtest-src"
  "/root/bmf/custom_deps/gtest-build"
  "/root/bmf/custom_deps/gtest-subbuild/gtest-populate-prefix"
  "/root/bmf/custom_deps/gtest-subbuild/gtest-populate-prefix/tmp"
  "/root/bmf/custom_deps/gtest-subbuild/gtest-populate-prefix/src/gtest-populate-stamp"
  "/root/bmf/custom_deps/gtest-subbuild/gtest-populate-prefix/src"
  "/root/bmf/custom_deps/gtest-subbuild/gtest-populate-prefix/src/gtest-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/root/bmf/custom_deps/gtest-subbuild/gtest-populate-prefix/src/gtest-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/root/bmf/custom_deps/gtest-subbuild/gtest-populate-prefix/src/gtest-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
