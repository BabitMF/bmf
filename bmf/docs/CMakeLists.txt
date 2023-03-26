cmake_minimum_required(VERSION 2.8)

project(bmf_doc)

set(BMF_DOXY_PROJECT_NAME             "BMF(Byte Multi-media Framework)")
set(BMF_DOXY_INPUT                    "\"${PROJECT_SOURCE_DIR}/\" \"${PROJECT_SOURCE_DIR}/example\" \
                                       \"${PROJECT_SOURCE_DIR}/../builder\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/dynamical_graph\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/c_module\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/edit\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/face_detect\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/generator\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/transcode\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/predict\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/pre_module\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/customize_module\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/run_by_config\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/server\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/subgraph\" \
                                       \"${PROJECT_SOURCE_DIR}/../example/sync_mode\" \
                                       \"${PROJECT_SOURCE_DIR}/../python_sdk/\" \
                                       \"${PROJECT_SOURCE_DIR}/../sdk/python_sdk/src/\" \
                                       \"${PROJECT_SOURCE_DIR}/../sdk/cpp_sdk/include/\" \
                                       \"${PROJECT_SOURCE_DIR}/../c_modules/include/\" \
                                       \"${PROJECT_SOURCE_DIR}/../c_modules/src/\"")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/bmf.conf.in
  ${CMAKE_CURRENT_BINARY_DIR}/bmf.conf
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/bmf_layout.xml.in
  ${CMAKE_CURRENT_BINARY_DIR}/bmf_layout.xml
)

add_custom_target(doc ALL
  COMMAND doxygen bmf.conf
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/images  ${CMAKE_CURRENT_BINARY_DIR}/html/images
)
