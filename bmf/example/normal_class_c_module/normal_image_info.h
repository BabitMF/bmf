#ifndef BMF_COPY_MODULE_H
#define BMF_COPY_MODULE_H

#include "bmf.h"

USE_BMF_SDK_NS

class NormalImageInfo : public Module
{
public:
    NormalImageInfo(Option option) : Module(option) { }

    ~NormalImageInfo() { }

    virtual void process(Task &task);
};

#endif

