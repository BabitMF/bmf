#include "connector_capi.h"


#define BMF_PROTECT(...) 	            \
	try{                                \
        __VA_ARGS__                     \
    } catch(const std::exception &e){   \
        s_bmf_last_error = e.what();    \
    }


using namespace bmf_sdk;


thread_local std::string s_bmf_last_error;
const char *bmf_engine_last_error()
{
    return s_bmf_last_error.c_str();
}


bmf_BMFGraph bmf_make_graph(char const *graph_json, bool is_path, bool need_merge) 
{
    BMF_PROTECT(
        return new bmf::BMFGraph(graph_json, is_path, need_merge);
    )
    return nullptr;
}

void bmf_graph_free(bmf_BMFGraph graph)
{
    BMF_PROTECT(
        if(graph){
            delete graph;
        }
    )
}

uint32_t bmf_graph_uid(bmf_BMFGraph graph)
{
    return graph->uid();
}

int bmf_graph_start(bmf_BMFGraph graph) 
{
    BMF_PROTECT(
        graph->start();
        return 0;
    )

    return -1;
}

int bmf_graph_close(bmf_BMFGraph graph)
{
    BMF_PROTECT(
        graph->close();
        return 0;
    )
    return -1;
}

int
bmf_graph_add_input_stream_packet(bmf_BMFGraph graph, char const *stream_name, bmf_Packet packet, bool block)
{
    BMF_PROTECT(
        return graph->add_input_stream_packet(stream_name, *packet, block);
    )
    return -1;
}

bmf_Packet
bmf_graph_poll_output_stream_packet(bmf_BMFGraph graph, char const *stream_name)
{
    BMF_PROTECT(
        return new bmf_sdk::Packet(graph->poll_output_stream_packet(stream_name));
    )
    return 0;
}

int bmf_graph_update(bmf_BMFGraph graph, char const *config, bool is_path) {
    BMF_PROTECT(
        graph->update(config, is_path);
        return 0;
    )
    return -1;
}

int bmf_graph_force_close(bmf_BMFGraph graph) {
    BMF_PROTECT(
        graph->force_close();
        return 0;
    )
    return -1;
}

char* bmf_graph_status(bmf_BMFGraph graph)
{
    BMF_PROTECT(
        return bmf_strdup(graph->status().jsonify().dump().c_str());
    )
    return 0;
}


bmf_BMFModule
bmf_make_module(char const *module_name, char const *option, char const *module_type, char const *module_path,
              char const *module_entry) {
    return new bmf::BMFModule(module_name, option, module_type, module_path, module_entry);
}

void bmf_module_free(bmf_BMFModule module) 
{
    if(module){
        delete module;
    }
}

int bmf_module_uid(bmf_BMFModule module)
{
    return module->uid();
}

int bmf_module_process(bmf_BMFModule module, bmf_Task task) 
{
    BMF_PROTECT(
        return module->process(*task);
    )
    return -1;
}

int bmf_module_init(bmf_BMFModule module) 
{
    BMF_PROTECT(
        return module->init();
    )
    return -1;
}

int bmf_module_reset(bmf_BMFModule module)
{
    BMF_PROTECT(
        return module->reset();
    )
    return -1;
}

int bmf_module_close(bmf_BMFModule module)
{
    BMF_PROTECT(
        return module->close();
    )
    return -1;
}
