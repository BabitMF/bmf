
if(FFMPEG_FOUND)
    return()
endif()

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
