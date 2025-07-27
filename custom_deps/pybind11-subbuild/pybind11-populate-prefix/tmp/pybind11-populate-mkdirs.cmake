# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/root/bmf/custom_deps/pybind11-src"
  "/root/bmf/custom_deps/pybind11-build"
  "/root/bmf/custom_deps/pybind11-subbuild/pybind11-populate-prefix"
  "/root/bmf/custom_deps/pybind11-subbuild/pybind11-populate-prefix/tmp"
  "/root/bmf/custom_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp"
  "/root/bmf/custom_deps/pybind11-subbuild/pybind11-populate-prefix/src"
  "/root/bmf/custom_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/root/bmf/custom_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/root/bmf/custom_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
