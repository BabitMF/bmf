
if(DEFINED ENV{GLOG_ROOT_PATH})
    set(GLOG_INCLUDE_DIR $ENV{GLOG_ROOT_PATH}/include)
    find_library(GLOG_LIBRARY glog HINTS $ENV{GLOG_ROOT_PATH}/lib)
else()
    find_path(GLOG_INCLUDE_DIR glog/logging.h)
    find_library(GLOG_LIBRARY glog)
endif()

if(GLOG_INCLUDE_DIR AND GLOG_LIBRARY)
    message("GLOG found: ${GLOG_LIBRARY}")
    add_library(glog INTERFACE IMPORTED GLOBAL)
    set_property(TARGET glog PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${GLOG_INCLUDE_DIR})
    set_property(TARGET glog PROPERTY INTERFACE_LINK_LIBRARIES
        ${GLOG_LIBRARY})

    if ($ENV{SCRIPT_EXEC_MODE} MATCHES "android")
        target_link_libraries(glog INTERFACE)
    else()
        target_link_libraries(glog INTERFACE pthread)
    endif()
else()
    message("GLOG not found in system path, using builtin module")
    add_subdirectory(3rd_party/glog)
endif()
