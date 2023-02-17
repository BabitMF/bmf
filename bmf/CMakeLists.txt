
add_subdirectory(sdk)
#add_subdirectory(go_module_sdk) #FIXME: compile failed
add_subdirectory(engine)
add_subdirectory(c_modules)
add_subdirectory(example/c_module)

if(NOT APPLE OR "${DEPLOYMENT_TARGET}" VERSION_GREATER_EQUAL "13.0" AND NOT ANDROID)
    add_subdirectory(example/cpp_builder)
    add_subdirectory(example/module_functor) # need filesystem support
endif()
add_subdirectory(cmd)
if(HMP_ENABLE_JNI)
    add_subdirectory(java)
endif()
if(BMF_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(IOS)
    add_subdirectory(object-c)
endif()

## Assemble BMF runtime(put all libs, incs and other resources under ${BMF_ASSEMBE_ROOT})
add_custom_target(bmf_assem ALL
    DEPENDS engine copy_module
    ${CMAKE_CURRENT_BINARY_DIR}/__always_update
)

add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/__always_update 
        ${CMAKE_CURRENT_BINARY_DIR}/always_update
    COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/always_update
)

function(copy_dir_to_bmf_module)
    if(ARGV1)
        set(TARGET_NAME ${ARGV1})
    else()
        set(TARGET_NAME ${ARGV0})
    endif()

    add_custom_command(TARGET bmf_assem
        POST_BUILD 
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                        ${CMAKE_CURRENT_SOURCE_DIR}/${ARGV0}
                        ${BMF_ASSEMBLE_ROOT}/bmf/${TARGET_NAME})
endfunction()

if(NOT IOS)
    copy_dir_to_bmf_module(builder)
    copy_dir_to_bmf_module(ffmpeg_engine)
    copy_dir_to_bmf_module(python_sdk)
    copy_dir_to_bmf_module(server)
    copy_dir_to_bmf_module(engine/c_engine/files files)
    copy_dir_to_bmf_module(example ../example)
    copy_dir_to_bmf_module(sdk/cpp_sdk/include include)
    copy_dir_to_bmf_module(engine/connector/include include)
else()
    copy_dir_to_bmf_module(sdk/cpp_sdk/include include)
endif()

#### 
if(ANDROID)
    add_custom_command(TARGET bmf_assem
        POST_BUILD
            COMMAND ${ANDROID_TOOLCHAIN_PREFIX}strip $<TARGET_FILE:hmp> 
    )
    if(BMF_ENABLE_JNI)
        add_custom_command(TARGET bmf_assem
            POST_BUILD
                COMMAND ${ANDROID_TOOLCHAIN_PREFIX}strip $<TARGET_FILE:_jhmp> 
        )
    endif()

    add_custom_command(TARGET bmf_assem
        POST_BUILD
            COMMAND ${ANDROID_TOOLCHAIN_PREFIX}strip $<TARGET_FILE:bmf_module_sdk> 
    )

    if(BMF_ENABLE_JNI)
        add_custom_command(TARGET bmf_assem
            POST_BUILD
                COMMAND ${ANDROID_TOOLCHAIN_PREFIX}strip $<TARGET_FILE:_jbmf> 
        )
    endif()
endif()



add_custom_command(TARGET bmf_assem
    POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy
                    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                    ${BMF_ASSEMBLE_ROOT}/bmf/)

add_custom_command(TARGET bmf_assem
    POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:copy_module> ${BMF_ASSEMBLE_ROOT}/example/c_module)

# json
add_custom_command(TARGET bmf_assem
    POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/3rd_party/json/include
                                                   ${BMF_ASSEMBLE_ROOT}/bmf/include)

# COPY hml related resources
set(HML_ROOT ${PROJECT_SOURCE_DIR}/bmf/hml)

add_custom_command(TARGET bmf_assem
    POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hmp> ${BMF_ASSEMBLE_ROOT}/bmf/lib
)

## copy symbol link of hmp and _jhmp
if(NOT APPLE AND NOT ANDROID)
    add_custom_command(TARGET bmf_assem
        POST_BUILD
            COMMAND cp -a 
            $<TARGET_FILE_DIR:hmp>/${CMAKE_SHARED_LIBRARY_PREFIX}hmp${CMAKE_SHARED_LIBRARY_SUFFIX} ${BMF_ASSEMBLE_ROOT}/bmf/lib
    )

    if(BMF_ENABLE_JNI)
        add_custom_command(TARGET bmf_assem
            POST_BUILD
                COMMAND cp -a 
                $<TARGET_FILE_DIR:_jhmp>/${CMAKE_SHARED_LIBRARY_PREFIX}_jhmp${CMAKE_SHARED_LIBRARY_SUFFIX} ${BMF_ASSEMBLE_ROOT}/bmf/lib
        )
    endif()
endif()


## includes
add_custom_command(TARGET bmf_assem
    POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${HML_ROOT}/include/ ${BMF_ASSEMBLE_ROOT}/bmf/include/
)

add_custom_command(TARGET bmf_assem
    POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${HML_ROOT}/third_party/fmt/include ${BMF_ASSEMBLE_ROOT}/bmf/include/
)

## python module
if(TARGET _hmp)
    add_custom_command(TARGET bmf_assem
        POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_hmp> ${BMF_ASSEMBLE_ROOT}/bmf/lib
    )
endif()

add_custom_command(TARGET bmf_assem
    POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${HML_ROOT}/hmp ${BMF_ASSEMBLE_ROOT}/hmp
)


if(IOS)
    if(BUILD_SHARED_LIBS)
        message(FATAL_ERROR "Only support static build for IOS platform")
    endif()

    set(IOS_ASSEMBLE_ROOT ${BMF_ASSEMBLE_ROOT}/ios)
    add_custom_command(TARGET bmf_assem
        POST_BUILD
        # create ios assemble root if needed
        COMMAND 
        ${CMAKE_COMMAND} -E make_directory ${IOS_ASSEMBLE_ROOT}
    )

    add_custom_command(TARGET bmf_assem
        POST_BUILD
        # archive all related static libraries to single one
        COMMAND
        libtool -static -o libbmf_module_sdk-all.a 
        $<TARGET_FILE:bmf_module_sdk>
        $<TARGET_FILE:hmp> 
        $<TARGET_FILE:fmt>
        $<TARGET_FILE:bmf_module_sdk_oc>
        $<TARGET_FILE:hmp_oc> 

        # copy all all public headers(cpp & object-c)
        COMMAND 
        ${CMAKE_COMMAND} -E copy_directory 
                            ${BMF_ASSEMBLE_ROOT}/bmf/include/bmf 
                            ${IOS_ASSEMBLE_ROOT}/include/bmf
        COMMAND 
        ${CMAKE_COMMAND} -E copy_directory 
                            ${BMF_ASSEMBLE_ROOT}/bmf/include/hmp
                            ${IOS_ASSEMBLE_ROOT}/include/hmp
        COMMAND 
        ${CMAKE_COMMAND} -E copy_directory 
                            ${BMF_ASSEMBLE_ROOT}/bmf/include/fmt
                            ${IOS_ASSEMBLE_ROOT}/include/fmt
        COMMAND 
        ${CMAKE_COMMAND} -E copy_directory 
                            ${BMF_ASSEMBLE_ROOT}/bmf/include/bmf_nlohmann
                            ${IOS_ASSEMBLE_ROOT}/include/bmf_nlohmann
        COMMAND 
        ${CMAKE_COMMAND} -E copy_directory 
                            ${CMAKE_CURRENT_SOURCE_DIR}/object-c/include/bmf/oc
                            ${IOS_ASSEMBLE_ROOT}/include/bmf/oc
        COMMAND 
        ${CMAKE_COMMAND} -E copy_directory 
                            ${HML_ROOT}/object-c/include/hmp/oc
                            ${IOS_ASSEMBLE_ROOT}/include/hmp/oc

        WORKING_DIRECTORY ${IOS_ASSEMBLE_ROOT} 
    )
endif()

if(IOS)
    add_subdirectory(example/static_module)
endif()
