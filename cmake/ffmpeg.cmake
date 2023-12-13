
if(FFMPEG_FOUND)
    return()
endif()

function(find_ffmpeg_version FFMPEG_VERSION_INCLUDE_DIR VERSION_VAR)

    find_file(FFMPEG_VERSION_FILE
            NAMES ffversion.h
            PATHS ${FFMPEG_VERSION_INCLUDE_DIR}
            PATH_SUFFIXES ffmpeg
            NO_DEFAULT_PATH)

    if(NOT FFMPEG_VERSION_FILE)
        message(FATAL_ERROR "Unable to find FFmpeg version file in ${FFMPEG_VERSION_INCLUDE_DIR}.")
    endif()


    file(STRINGS ${FFMPEG_VERSION_FILE} FFMPEG_VERSION_DEFINE
        REGEX "#define FFMPEG_VERSION ")


    string(REGEX MATCH "FFMPEG_VERSION \"([^\"]+)\"" FFMPEG_VERSION_STRING ${FFMPEG_VERSION_DEFINE})

    set(FFMPEG_VERSION_MATCHED ${CMAKE_MATCH_1})

    if(FFMPEG_VERSION_MATCHED)
        string(REGEX MATCHALL "([0-9]+)" VERSION_PARTS ${FFMPEG_VERSION_MATCHED})
        list(GET VERSION_PARTS 0 VERSION_MAJOR)
        list(GET VERSION_PARTS 1 VERSION_MINOR)
        # 拼接主版本号和次版本号
        set(FFMPEG_VERSION_CLEAN "${VERSION_MAJOR}${VERSION_MINOR}")
        set(${VERSION_VAR} ${FFMPEG_VERSION_CLEAN} PARENT_SCOPE)
        if(FFMPEG_VERSION_CLEAN VERSION_LESS 40 OR FFMPEG_VERSION_CLEAN VERSION_GREATER 51)
            message(FATAL_ERROR "ffmpeg version is less than 4.0 or greater than 5.1, which is not supported.")
        endif()
        set(${VERSION_VAR} ${FFMPEG_VERSION_CLEAN} PARENT_SCOPE)
        message(STATUS "Detected FFmpeg version: ${FFMPEG_VERSION_CLEAN}")
    else()
        message(FATAL_ERROR "Unable to parse FFmpeg version.")
    endif()
endfunction()


if(DEFINED ENV{FFMPEG_ROOT_PATH})
    set(FFMPEG_LIBRARY_DIR $ENV{FFMPEG_ROOT_PATH}/lib)
    set(FFMPEG_INCLUDE_DIR $ENV{FFMPEG_ROOT_PATH}/include)
else()
    find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
            HINTS /opt/conda /usr/
            PATH_SUFFIXES ffmpeg)
    find_library(FFMPEG_LIBRARY_DIR avcodec
            HINTS /usr/)
endif()

if(FFMPEG_LIBRARY_DIR AND FFMPEG_INCLUDE_DIR)
    message(STATUS "find FFmpeg, FFMPEG_INCLUDE_DIR:" ${FFMPEG_INCLUDE_DIR} ", FFMPEG_LIBRARY_DIR:" ${FFMPEG_LIBRARY_DIR})
    function(define_ffmpeg_target)
        find_library(LIBRARY ${ARGV0} HINTS ${FFMPEG_LIBRARY_DIR})
        if(LIBRARY)
            add_library(ffmpeg::${ARGV0} INTERFACE IMPORTED GLOBAL)
            set_property(TARGET ffmpeg::${ARGV0} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                ${FFMPEG_INCLUDE_DIR})
            set_property(TARGET ffmpeg::${ARGV0} PROPERTY INTERFACE_LINK_LIBRARIES
                ${LIBRARY})
            unset(LIBRARY CACHE)
        endif()
    endfunction()

    define_ffmpeg_target(avcodec)
    define_ffmpeg_target(avformat)
    define_ffmpeg_target(avfilter)
    define_ffmpeg_target(avutil)
    define_ffmpeg_target(avdevice)
    define_ffmpeg_target(swscale)
    define_ffmpeg_target(postproc)
    define_ffmpeg_target(swresample)

    if(TARGET ffmpeg::avcodec)
        find_ffmpeg_version(${FFMPEG_INCLUDE_DIR}/libavutil BMF_FFMPEG_VERSION)
        add_definitions(-DBMF_FFMPEG_VERSION=${BMF_FFMPEG_VERSION})
        set(FFMPEG_FOUND TRUE)
    else()
        set(FFMPEG_FOUND FALSE)
    endif()
else()
    find_package(PkgConfig REQUIRED)
    function(define_ffmpeg_target)
        pkg_check_modules(LIBRARY lib${ARGV0})
        if(LIBRARY_FOUND)
            add_library(ffmpeg::${ARGV0} INTERFACE IMPORTED GLOBAL)
            set_property(TARGET ffmpeg::${ARGV0} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                ${LIBRARY_INCLUDE_DIRS})
            set_property(TARGET ffmpeg::${ARGV0} PROPERTY INTERFACE_LINK_LIBRARIES
                ${LIBRARY_LINK_LIBRARIES})
            unset(LIBRARY CACHE)
            if("${ARGV0}" STREQUAL "avutil")
                find_ffmpeg_version(${LIBRARY_INCLUDE_DIRS}/libavutil BMF_FFMPEG_VERSION)
                add_definitions(-DBMF_FFMPEG_VERSION=${BMF_FFMPEG_VERSION})
            endif()
        endif()
    endfunction()

    define_ffmpeg_target(avcodec)
    define_ffmpeg_target(avformat)
    define_ffmpeg_target(avfilter)
    define_ffmpeg_target(avutil)
    define_ffmpeg_target(avdevice)
    define_ffmpeg_target(swscale)
    define_ffmpeg_target(postproc)
    define_ffmpeg_target(swresample)

    if(TARGET ffmpeg::avcodec)
        set(FFMPEG_FOUND TRUE)
    else()
        set(FFMPEG_FOUND FALSE)
    endif()
endif()
