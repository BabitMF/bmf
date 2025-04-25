set(Libuuid_INCLUDE_DIRS "")
set(Libuuid_LIBRARIES "")

include(FetchContent)

set(LIBUUID_STATIC ON CACHE BOOL "enable static build for libuuid" FORCE)
set(LIBUUID_SHARED OFF CACHE BOOL "diable shared build for libuuid" FORCE)
FetchContent_Declare(libuuid-cmake
    GIT_REPOSITORY  https://github.com/gershnik/libuuid-cmake.git
    GIT_TAG         v2.39.1
    GIT_SHALLOW     TRUE
)
FetchContent_MakeAvailable(libuuid-cmake)

if(NOT TARGET uuid_static)
    message(FATAL_ERROR "libuuid-cmake target fetch failed")
else()
    target_compile_options(uuid_static PRIVATE -fPIC)
    set(Libuuid_LIBRARIES uuid::uuid)
    get_target_property(Libuuid_INCLUDE_DIRS uuid_static INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT EXISTS ${Libuuid_INCLUDE_DIRS}/uuid/uuid.h)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy
                ${FETCHCONTENT_BASE_DIR}/util-linux-src/libuuid/src/uuid.h  # 源文件路径
                ${Libuuid_INCLUDE_DIRS}/uuid/uuid.h  # 目标路径
            RESULT_VARIABLE copy_result
            ERROR_VARIABLE copy_error
        )
    endif()
    set(Libuuid_FOUND YES)
endif()

