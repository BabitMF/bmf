
#include <bmf/sdk/task.h>
#include <bmf/sdk/module_functor.h>

namespace bmf_sdk{


struct ModuleFunctor::Private
{
    std::shared_ptr<Module> module;
    std::vector<int> iids;
    std::vector<int> oids;
    std::vector<bool> eofs;

    Task task;
    ~Private(){
        try{
            module->close();
        }
        catch(std::exception &e){
            HMP_WRN("Exception {} when do module close", e.what());
        }
    }
};

ModuleFunctor::ModuleFunctor(const std::shared_ptr<Module> &m, int ninputs, int noutputs)
{
    HMP_REQUIRE(m, "Null module ptr detected");
    HMP_REQUIRE(ninputs >= 0, "Invalid ninputs = {}", ninputs);
    HMP_REQUIRE(noutputs >= 0, "Invalid noutputs = {}", noutputs);

    // module init
    auto rc = m->init();
    HMP_REQUIRE(rc == 0, "Module inital failed with rc={}", rc);

    self = std::make_shared<Private>();
    self->module = m;
    for(int i = 0; i < ninputs; ++i){
        self->iids.push_back(i);
    }
    for(int i = 0; i < noutputs; ++i){
        self->oids.push_back(i);
        self->eofs.push_back(false);
    }

    self->task = Task(self->module->node_id_, self->iids, self->oids);
}


ModuleFunctor::~ModuleFunctor()
{
}


ModuleFunctor& ModuleFunctor::execute(const std::vector<Packet> &inputs, bool cleanup)
{
    HMP_REQUIRE(inputs.size() == self->iids.size(),
             "Expect {} inputs, got {} inputs", self->iids.size(), inputs.size());
    for(size_t i = 0; i < self->iids.size(); ++i){
        if(inputs[i]){
            self->task.fill_input_packet(self->iids[i], inputs[i]);
        }
    }

    //
    if(self->task.timestamp() == DONE){
        throw ProcessDone("Task done");
    }

    if(cleanup){
        // clear all un-fetch results
        for(auto &it : self->task.get_outputs()){
            while(!it.second->empty()){
                it.second->pop();
            }
        }
    }

    auto rc = self->module->process(self->task);
    if(rc != 0){
        throw std::runtime_error(fmt::format("Process failed with error {}", rc));
    }

    if(self->task.timestamp() == DONE){
        for(size_t i = 0; i < self->oids.size(); ++i){
            if(!self->task.output_queue_empty(self->oids[i])){
                return *this;
            }
        }

        // all output queue is empty
        throw ProcessDone("Task done");
    }

    return *this;
}


std::vector<Packet> ModuleFunctor::fetch(int idx)
{
    std::vector<Packet> pkts;
    auto oid = self->oids[idx];
    while(!self->task.output_queue_empty(oid)){
        Packet pkt;
        self->task.pop_packet_from_out_queue(oid, pkt);
        if(pkt && pkt.timestamp() == BMF_EOF){
            self->eofs[idx] = true;
            break;
        }
        
        pkts.push_back(pkt);
    }

    if(self->eofs[idx] && pkts.size() == 0){
        throw ProcessDone("Receive EOF packet");
    }

    return pkts;
}


std::vector<Packet> ModuleFunctor::operator()(const std::vector<Packet> &inputs)
{
    execute(inputs);

    std::vector<Packet> outputs;
    int neof = 0;
    for(size_t i = 0; i < self->oids.size(); ++i){
        if(self->eofs[i]){
            neof += 1;
            continue;
        }

        //
        std::vector<Packet> opkts;
        try{
            opkts = fetch(i);
        }
        catch(ProcessDone &e){
            self->eofs[i] = true;
            neof += 1;
            continue;
        }

        //
        HMP_REQUIRE(opkts.size() <= 1, 
                    "ModuleFunctor: more than one output packet is not supported, got {}", opkts.size());
        if(opkts.size()){
            outputs.push_back(opkts[0]);
        }
        else{
            outputs.push_back(Packet());
        }
    }

    if(neof == self->oids.size() && self->oids.size() > 0){
        throw ProcessDone("Receive EOF packet");
    }

    return outputs;
}


Module &ModuleFunctor::module() const
{
    return *self->module;
}


int ModuleFunctor::ninputs() const
{
    return self->iids.size();
}

int ModuleFunctor::noutputs() const
{
    return self->oids.size();
}


ModuleFunctor make_sync_func(const ModuleInfo &info, int32_t ninputs, int32_t noutputs,
                             const JsonParam &option, int32_t node_id)
{
    auto &M = ModuleManager::instance();
    auto factory = M.load_module(info);
    if(factory == nullptr){
        throw std::runtime_error("Load module " + info.module_name + " failed");
    }

    return ModuleFunctor(factory->make(node_id, option), ninputs, noutputs);
}



} //namespace bmf_sdk