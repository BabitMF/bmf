# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/root/bmf/custom_deps/dlpack-src"
  "/root/bmf/custom_deps/dlpack-build"
  "/root/bmf/custom_deps/dlpack-subbuild/dlpack-populate-prefix"
  "/root/bmf/custom_deps/dlpack-subbuild/dlpack-populate-prefix/tmp"
  "/root/bmf/custom_deps/dlpack-subbuild/dlpack-populate-prefix/src/dlpack-populate-stamp"
  "/root/bmf/custom_deps/dlpack-subbuild/dlpack-populate-prefix/src"
  "/root/bmf/custom_deps/dlpack-subbuild/dlpack-populate-prefix/src/dlpack-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/root/bmf/custom_deps/dlpack-subbuild/dlpack-populate-prefix/src/dlpack-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/root/bmf/custom_deps/dlpack-subbuild/dlpack-populate-prefix/src/dlpack-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
