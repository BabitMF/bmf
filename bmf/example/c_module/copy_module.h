#ifndef BMF_COPY_MODULE_H
#define BMF_COPY_MODULE_H

#include <bmf/sdk/bmf.h>
#include <bmf/sdk/packet.h>

USE_BMF_SDK_NS

class CopyModule : public Module
{
public:
    CopyModule(int node_id,JsonParam option) : Module(node_id,option) { }

    ~CopyModule() { }

    virtual int process(Task &task);
};

#endif

