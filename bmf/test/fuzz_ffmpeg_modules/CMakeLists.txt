file(GLOB SRCS *.cpp)

if(FUZZTEST_ENABLE_FUZZING_MODE)
    fuzztest_setup_fuzzing_flags()
endif()

add_executable(fuzz_ffmpeg_modules ${SRCS})

target_link_libraries(fuzz_ffmpeg_modules
   PRIVATE
       builtin_modules engine bmf_module_sdk
       gtest ${BMF_FFMPEG_TARGETS}
)

if(BMF_ENABLE_FUZZTEST)
    link_fuzztest(fuzz_ffmpeg_modules)
endif()

target_link_libraries(fuzz_ffmpeg_modules PRIVATE gtest_main)

install(TARGETS fuzz_ffmpeg_modules
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)
