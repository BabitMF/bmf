#pragma once

#ifdef __APPLE__
#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED < 130000
#define CUSTOM_FILESYSTEM
#endif //__IPHONE_OS_VERSION_MIN_REQUIRED
#endif //__APPLE__

#ifdef ANDROID
#define CUSTOM_FILESYSTEM
#endif //ANDROID

#ifndef CUSTOM_FILESYSTEM
#include <filesystem>
namespace fs = std::filesystem;

#else //CUSTOM_FILESYSTEM


#include <string>
#include <unistd.h>
#include <sys/stat.h>

namespace fs{

// simple std::filesystem::path implementation(api compat)
// remove it when IOS version upgrade to 13.0 or above
class path
{
    std::string p_;
public:
    path(const std::string &p) : p_(p)
    {
    }

    path operator/(const path &p)
    {
        return p_ + "/" + p.p_;
    }

    path& operator/=(const path &p)
    {
        p_ = p_ + "/" + p.p_;
	    return *this;
    }

    operator std::string() const
    {
        return p_;
    }

    std::string string() const
    {
        return p_;
    }

    std::string extension() const
    {
        size_t pos = p_.find_last_of(".");
        if (pos == std::string::npos)
            return "";
        return p_.substr(pos);
    }

    path replace_extension(const std::string &ext)
    {
        size_t pos = p_.find_last_of(".");
        if (pos == std::string::npos)
            return p_ + ext;
        return p_.substr(0, pos) + ext;
    }

    path parent_path() const {
        return p_.substr(0, p_.find_last_of("/\\"));
    }

    path filename() const {
        auto p = p_.find_last_of("/\\");
        if(p == std::string::npos){
            return p_;
        }
        else{
            return p_.substr(p+1);
        }
    }
};


static bool exists(const std::string &p)
{
    struct stat sb;
    return stat(p.c_str(), &sb) == 0;
}

static bool is_directory(const std::string &p)
{
    struct stat sb;
    if (stat(p.c_str(), &sb))
        return false;
    return S_ISDIR(sb.st_mode);
}


static path current_path()
{
    char tmp[4096];
    if(::getcwd(tmp, 4096) == NULL){
        throw std::runtime_error("Internal error in getcwd():" + std::string(strerror(errno)));
    }
    return path(tmp);
}


static path operator/(const path &a, const path &b)
{
    path ret(a);
    ret /= b;
    return ret;
}


} //namespace fs

#endif //CUSTOM_FILESYSTEM
