#ifndef BMF_COPY_MODULE_H
#define BMF_COPY_MODULE_H

#include "bmf.h"

USE_BMF_SDK_NS

class CopyModule : public Module
{
public:
    CopyModule(Option option) : Module(option) { }

    ~CopyModule() { }

    virtual void process(Task &task);
};

#endif

