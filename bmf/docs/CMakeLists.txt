cmake_minimum_required(VERSION 2.8)

project(bmf_doc)

set(BMF_DOXY_PROJECT_NAME             "BMF(Babit Multimedia Framework)")
set(BMF_DOXY_INPUT                    "\"${PROJECT_SOURCE_DIR}/\" \
                                       \"${PROJECT_SOURCE_DIR}/../builder\" \
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
