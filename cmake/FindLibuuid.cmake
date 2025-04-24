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
if(NOT TARGET uuid_static)  # 通过目标存在性验证[7](@ref)
    message(FATAL_ERROR "libuuid-cmake target fetch failed")
else()
    target_compile_options(uuid_static PRIVATE -fPIC)
    set(Libuuid_LIBRARIES uuid::uuid)
    get_target_property(Libuuid_INCLUDE_DIRS uuid_static INTERFACE_INCLUDE_DIRECTORIES)
    set(Libuuid_FOUND YES)
endif()

