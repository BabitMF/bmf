# Execute graph with graph config json file

file(GLOB SRCS src/run_graph.cpp)

add_executable(run_bmf_graph ${SRCS})


if(BMF_USE_MEDIACODEC)
    target_link_libraries(run_bmf_graph
        PRIVATE 
        nlohmann engine bmf_module_sdk
        -pie -fPIE -Wl,--export-dynamic
    )
else()
    target_link_libraries(run_bmf_graph
        PRIVATE 
        nlohmann engine bmf_module_sdk
    )
endif()

# Trace format log tool

file(GLOB TRACE_LOG_SRCS src/trace_format_log.cpp)

add_executable(trace_format_log ${TRACE_LOG_SRCS})

target_link_libraries(trace_format_log
    PRIVATE 
    bmf_module_sdk
)

# module manager
file(GLOB MODULE_MANAGER_SRCS src/module_manager.cpp)

add_executable(module_manager ${MODULE_MANAGER_SRCS})

target_link_libraries(module_manager
    PRIVATE
    engine
    bmf_module_sdk
)

# Build targets
    
install(TARGETS run_bmf_graph trace_format_log module_manager
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Tool suite
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(CURSES_NCURSESW IMPORTED_TARGET ncursesw)
endif()

# only compile tool suite if ncursesw is detected
# ncursesw can be installed by command line "apt-get install libncursesw5-dev" on Debian
if(CURSES_NCURSESW_FOUND)
    file(GLOB SUITE_LOG_SRCS
        include/menu.h
        include/trace_config_menu.h
        src/menu.cpp
        src/suite.cpp
    )

    add_executable(bmf_suite ${SUITE_LOG_SRCS})

    set_target_properties(bmf_suite PROPERTIES OUTPUT_NAME "bmf")

    target_link_libraries(bmf_suite PUBLIC PkgConfig::CURSES_NCURSESW)

    install(TARGETS bmf_suite
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
    )
else()
    message("Skipping BMF tool suite compilation as ncursesw is not detected")
endif()
