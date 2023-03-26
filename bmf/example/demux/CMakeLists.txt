cmake_minimum_required(VERSION 3.12)
project(pull_stream)
# language setting
# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../install/c_modules)
# boost
find_package(Boost REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})
message(${Boost_INCLUDE_DIRS})
message(${Boost_LIBRARY_DIRS})

set (bmf_dir /opt/tiger/bmf)

# include dir
set(MODULE_INCLUDE_PATH

        ${PROJECT_SOURCE_DIR}/../../install/module_sdk/include
        ${PROJECT_SOURCE_DIR}/../../../3rd_party/ffmpeg_bin/linux/build/include
        ${PROJECT_SOURCE_DIR}/../../../3rd_party/json/include
        /usr/local/include)

message(${MODULE_INCLUDE_PATH})
LINK_DIRECTORIES("/usr/local/lib")
include_directories(${MODULE_INCLUDE_PATH})
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/../../install/module_sdk/lib)
LINK_LIBRARIES(bmf_module_sdk)


include_directories(/usr/include/python3.7m)
LINK_DIRECTORIES(/usr/local/lib /usr/lib/python3.7/config-3.7m-x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/)
LINK_LIBRARIES(boost_numpy37 python3.7 boost_python37 stdc++fs boost_system pthread dl)

SET(MODULE_SRCS ${PROJECT_SOURCE_DIR}/pull_stream.cpp)

add_library(${PROJECT_NAME} SHARED ${MODULE_SRCS})
