/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../include/common.h"
#include "../include/graph.h"
#include <bmf/sdk/log.h>

#include "gtest/gtest.h"

#include <fstream>

USE_BMF_ENGINE_NS
USE_BMF_SDK_NS
TEST(graph, start) {
    bmf_nlohmann::json graph_json;
    std::string config_file = "../files/graph_start.json";
    std::ifstream gs(config_file);
    gs >> graph_json;
    GraphConfig graph_config(graph_json);
    std::map<int, std::shared_ptr<Module> > pre_modules;
    std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings;
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules,callback_bindings);
    graph->start();
    graph->close();

}

void SignalHandle2(const char *data, int size) {
    std::ofstream fs("glog_dump.log", std::ios::app);
    std::string str = std::string(data, size);
    fs << str;
    fs.close();
    
    BMFLOG(BMF_ERROR) << str;
}
//TEST(graph, decode){
////    google::InitGoogleLogging("main");
//    google::SetStderrLogging(google::INFO);
//    google::InstallFailureSignalHandler();
//    google::InstallFailureWriter(&SignalHandle2);
//    setenv("PYTHONPATH",
//           "../../../3rd_party/pyav/:.:../python_sdk",
//           1);
//    Py_Initialize();
//
////
//    time_t time1=clock();
//    std::string config_file = "../files/decode_graph.json";
//    GraphConfig graph_config(config_file);
//    std::map<int, std::shared_ptr<Module> > pre_modules;
//    std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules);
//    std::cout<<"init graph success"<<std::endl;
//    PyEval_InitThreads();
//    PyEval_ReleaseThread(PyThreadState_Get());
//    graph->start();
//    graph->close();
//    time_t time2 = clock();
//    std::cout<<"time:"<<time2-time1<<std::endl;
//    PyGILState_Ensure();
//}
TEST(graph, decode_encode) {
    BMFLOG_SET_LEVEL(BMF_INFO);

    time_t time1 = clock();
    bmf_nlohmann::json graph_json;
    std::string config_file = "../files/graph.json";
    std::ifstream gs(config_file);
    gs >> graph_json;
    GraphConfig graph_config(graph_json);
    std::map<int, std::shared_ptr<Module> > pre_modules;
    std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings;
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules,callback_bindings);
    std::cout << "init graph success" << std::endl;

    graph->start();
    graph->close();
    time_t time2 = clock();
    std::cout << "time:" << time2 - time1 << std::endl;

}

TEST(graph, c_decode_encode) {
    BMFLOG_SET_LEVEL(BMF_INFO);

    time_t time1 = clock();
    bmf_nlohmann::json graph_json;
    std::string config_file = "../files/graph_c.json";
    std::ifstream gs(config_file);
    gs >> graph_json;
    GraphConfig graph_config(graph_json);
    std::map<int, std::shared_ptr<Module> > pre_modules;
    std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings;
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules,callback_bindings);
    std::cout << "init graph success" << std::endl;

//    PyEval_ReleaseThread(PyThreadState_Get());
    graph->start();
    graph->close();
    time_t time2 = clock();
    std::cout << "time:" << time2 - time1 << std::endl;

}

TEST(graph, dynamic_add) {
    BMFLOG_SET_LEVEL(BMF_INFO);

    time_t time1 = clock();
    std::string config_file = "../files/graph_dyn.json";
    std::string dyn_config_file = "../files/dynamic_add.json";
    GraphConfig graph_config(config_file);
    GraphConfig dyn_config(dyn_config_file);
    std::map<int, std::shared_ptr<Module> > pre_modules;
    std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings;
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules, callback_bindings);
    std::cout << "init graph success" << std::endl;

    graph->start();
    usleep(400000);

    std::cout << "graph dynamic add nodes" << std::endl;
    graph->update(dyn_config);

    graph->close();
    time_t time2 = clock();
    std::cout << "time:" << time2 - time1 << std::endl;

}

TEST(graph, dynamic_remove) {
    BMFLOG_SET_LEVEL(BMF_INFO);

    time_t time1 = clock();
    std::string config_file = "../files/graph_passthru.json";
    std::string dyn_add_config_file = "../files/dynamic_passthru.json";
    std::string dyn_remove_config_file = "../files/dynamic_remove.json";
    GraphConfig graph_config(config_file);
    GraphConfig dyn_add_config(dyn_add_config_file);
    GraphConfig dyn_remove_config(dyn_remove_config_file);
    std::map<int, std::shared_ptr<Module> > pre_modules;
    std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings;
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules, callback_bindings);
    std::cout << "init graph success" << std::endl;

    graph->start();
    usleep(10000);

    std::cout << "graph dynamic add nodes" << std::endl;
    graph->update(dyn_add_config);
    usleep(10000);

    std::cout << "graph dynamic remove nodes" << std::endl;
    graph->update(dyn_remove_config);
    sleep(2);

    graph->force_close();//here only the pass_through_module lefted
    time_t time2 = clock();
    std::cout << "time:" << time2 - time1 << std::endl;

}

TEST(graph, decode_filter_encode) {
    BMFLOG_SET_LEVEL(BMF_INFO);
    bmf_nlohmann::json graph_json;
    std::string config_file = "../files/filter_opt_graph.json";
    std::ifstream gs(config_file);
    gs >> graph_json;
    GraphConfig graph_config(graph_json);
    std::map<int, std::shared_ptr<Module> > pre_modules;
    std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings;
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules,callback_bindings);
    graph->start();
    graph->close();
}

TEST(graph, c_decode_filter_encode) {
    BMFLOG_SET_LEVEL(BMF_INFO);

    bmf_nlohmann::json graph_json;
    std::string config_file = "../files/filter_opt_graph_c.json";
    std::ifstream gs(config_file);
    gs >> graph_json;
    GraphConfig graph_config(graph_json);
    std::map<int, std::shared_ptr<Module> > pre_modules;
    std::map<int, std::shared_ptr<ModuleCallbackLayer> > callback_bindings;
    std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules,callback_bindings);

    graph->start();
    graph->close();
}
//TEST(graph, multi_process) {
//    google::InitGoogleLogging("main");
//    google::SetStderrLogging(google::INFO);
//    google::InstallFailureSignalHandler();
//    google::InstallFailureWriter(&SignalHandle2);
//
//    for (int i = 1; i <= 100; i++) {
//        std::cout<<"********test time:"<<i<<std::endl;
//        LOG(ERROR)<<"start";
//        std::string config_file = "../files/multi.json";
//        GraphConfig graph_config(config_file);
//        std::map<int, std::shared_ptr<Module> > pre_modules;
//        std::shared_ptr<Graph> graph = std::make_shared<Graph>(graph_config, pre_modules);
//        graph->start();
//        graph->close();
//        std::cout<<"close graph*******"<<std::endl;
////        sleep(10);
//        LOG(ERROR)<<"end"<<std::endl;
//    }
//    google::ShutdownGoogleLogging();
//}
