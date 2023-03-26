
if(FFMPEG_FOUND)
    return()
endif()

if(DEFINED ENV{FFMPEG_ROOT_PATH})
    if (${APPLE})
        set(FFMPEG_ROOT $ENV{FFMPEG_ROOT_PATH}/lib)
        set(FFMPEG_INCLUDE_DIR $ENV{FFMPEG_ROOT_PATH}/include)
    else()
        set(FFMPEG_ROOT $ENV{FFMPEG_ROOT_PATH})
    endif()
else()
    find_path(FFMPEG_ROOT include/libavcodec/avcodec.h 
            HINTS /opt/conda /usr/)
    set(FFMPEG_INCLUDE_DIR ${FFMPEG_ROOT}/include)
endif()


if(FFMPEG_ROOT)
    set(FFMPEG_INCLUDE_DIR ${FFMPEG_ROOT}/include)

    function(define_ffmpeg_target)
        find_library(LIBRARY ${ARGV0} HINTS ${FFMPEG_ROOT}/lib ${FFMPEG_ROOT})
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
    set(FFMPEG_FOUND FALSE)
endif()
