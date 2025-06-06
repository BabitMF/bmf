
if ($ENV{SCRIPT_EXEC_MODE} MATCHES "android")
    set(GOOGLETEST_ROOT $ENV{ANDROID_NDK_ROOT}/sources/third_party/googletest)
    add_library(gtest STATIC ${GOOGLETEST_ROOT}/src/gtest_main.cc ${GOOGLETEST_ROOT}/src/gtest-all.cc)
    target_include_directories(gtest PRIVATE ${GOOGLETEST_ROOT})
    target_include_directories(gtest PUBLIC ${GOOGLETEST_ROOT}/include)
else()
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
    find_library(GTEST_LIBRARY gtest)
    if(${APPLE})
        find_library(GTEST_MAIN_LIBRARY gtest_main)
    endif()

    if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY)
        message("GTEST found: ${GTEST_LIBRARY}")
        add_library(gtest INTERFACE IMPORTED GLOBAL)
        set_property(TARGET gtest PROPERTY INTERFACE_INCLUDE_DIRECTORIES
            ${GTEST_INCLUDE_DIR})
        set_property(TARGET gtest PROPERTY INTERFACE_LINK_LIBRARIES
            ${GTEST_LIBRARY})
        target_link_libraries(gtest INTERFACE pthread)
        if(${APPLE})
            add_library(gtest_main INTERFACE IMPORTED GLOBAL)
            set_property(TARGET gtest_main PROPERTY INTERFACE_LINK_LIBRARIES
                ${GTEST_MAIN_LIBRARY})
        endif()
    else()
        message("GTEST not found in system path, fetching via FetchContent to ${PROJECT_SOURCE_DIR}/3rd_party/gtest")        
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
            SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rd_party/gtest
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()
