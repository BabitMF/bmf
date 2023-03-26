
if(FFMPEG_FOUND)
    return()
endif()

find_path(FFMPEG_ROOT include/libavcodec/avcodec.h
     HINTS /opt/conda /usr/)

if(FFMPEG_ROOT)
    set(FFMPEG_INCLUDE_DIR ${FFMPEG_ROOT}/include)

    function(define_ffmpeg_target)
        find_library(LIBRARY ${ARGV0} HINTS ${FFMPEG_ROOT}/lib ${FFMPEG_ROOT})
        add_library(ffmpeg::${ARGV0} INTERFACE IMPORTED GLOBAL)
        set_property(TARGET ffmpeg::${ARGV0} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
            ${FFMPEG_INCLUDE_DIR})
        set_property(TARGET ffmpeg::${ARGV0} PROPERTY INTERFACE_LINK_LIBRARIES
            ${LIBRARY})
        unset(LIBRARY CACHE)
    endfunction()

    define_ffmpeg_target(avcodec)
    define_ffmpeg_target(avformat)
    define_ffmpeg_target(avfilter)
    define_ffmpeg_target(avutil)
    define_ffmpeg_target(avdevice)
    define_ffmpeg_target(swscale)
    define_ffmpeg_target(postproc)
    define_ffmpeg_target(swresample)

    set(FFMPEG_FOUND TRUE)
else()
    set(FFMPEG_FOUND FALSE)
endif()
