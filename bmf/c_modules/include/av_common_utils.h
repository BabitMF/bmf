#include <string>
#include <libavutil/error.h>

inline std::string error_msg(int err) {
    char errbuf[128];
    const char *errbuf_ptr = errbuf;
    if (av_strerror(err, errbuf, sizeof(errbuf)) < 0)
        errbuf_ptr = strerror(AVUNERROR(err));
    std::string estr = errbuf_ptr;
    return estr;
}
