/*
 * Copyright 2023 Babit Authors
 *
 * This file is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 */

#include "gtest/gtest.h"
#include "python_module.h"
#include <fstream>
#include <istream>

USE_BMF_SDK_NS

void dump_node_json(JsonParam &json_param) {
    std::cout << "********node json:" << json_param.dump() << std::endl;
}

void create_python_module(JsonParam &node_param, std::shared_ptr<PythonModule> &python_module) {
    std::string module_name;
    dump_node_json(node_param);
    node_param.get_string("module", module_name);
    JsonParam node_option_json;
    node_param.get_object("option", node_option_json);
    python_module = std::make_shared<PythonModule>(module_name, module_name, 0, node_option_json);
}

void get_python_module_info(JsonParam &node_param, std::string &module_name, std::string &option) {
//    dump_node_json(node_param);
    node_param.get_string("module", module_name);
    JsonParam node_option_json;
    node_param.get_object("option", node_option_json);
    option = node_option_json.dump();
}

void create_python_module_from_string(std::string module_name, std::string option,
                                      std::shared_ptr<PythonModule> &python_module) {
    std::istringstream s;
    bmf_nlohmann::json opt;
    s.str(option);
    s >> opt;
    python_module = std::make_shared<PythonModule>(module_name, module_name, 0, JsonParam(opt));
}

TEST(python_module, ffmpeg_module) {

    setenv("PYTHONPATH",
           "../../../3rd_party/pyav/:.:../../python_modules:../../python_module_sdk",
           1);
    Py_Initialize();
    JsonParam graph_config;
    graph_config.load("../files/graph.json");
    std::vector<JsonParam> node_jsons;
    graph_config.get_object_list("nodes", node_jsons);
    std::shared_ptr<PythonModule> python_module;
    std::cout<<node_jsons[0].dump()<<std::endl;
    create_python_module(node_jsons[0], python_module);
    std::vector<int> input_labels;
    std::vector<int> output_labels;
    output_labels.push_back(0);
    Task task = Task(0, input_labels, output_labels);

    python_module->process(task);
}

void thread_decode(std::shared_ptr<PythonModule> decoder_module2) {
//    setenv("PYTHONPATH",
//           "/Users/bytedance/Project/company/python3/bmf_python3/bmf/3rd_party/pyav:.:/Users/bytedance/Project/company/python2/bmf_master/bmf/bmf/c_engine/python_sdk",
//           1);
//    Py_Initialize();
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    boost::python::object python_file = boost::python::import("timestamp");
    sleep(1);
//    PyGILState_Release(gstate);
//    sleep(10);
//    gstate = PyGILState_Ensure();

    std::string decoder_name;
    std::string decoder_option;
    std::string encoder_name;
    std::string encoder_option;
    JsonParam graph_config;
    graph_config.load("../files/graph.json");
    std::vector<JsonParam> node_jsons;
    graph_config.get_object_list("nodes", node_jsons);
    std::shared_ptr<PythonModule> decoder_module;
    get_python_module_info(node_jsons[0], decoder_name, decoder_option);
    get_python_module_info(node_jsons[1], encoder_name, encoder_option);
    create_python_module_from_string(decoder_name, decoder_option, decoder_module);

    bool decoder_finished = false;
//    while (true) {
//
//        Task task = Task(0, 0, 2);
//        std::cout << "decoder_module start" << std::endl;
//        if (not decoder_finished) {
//            decoder_module->process(task);
//        }
//        std::cout << "decoder_module end" << std::endl;
//        if (task.get_timestamp() == DONE) {
//            decoder_finished = true;
//            break;
//        }
////        sleep(1);
//    }
    /* Release the thread. No Python API allowed beyond this point. */
    PyGILState_Release(gstate);
}

void test() {
    sleep(3);
    boost::python::object python_file = boost::python::import("packet");
    boost::python::object packet = python_file.attr("Packet")();
}

void thread_sleep(std::shared_ptr<PythonModule> samples) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    test();
    /* Release the thread. No Python API allowed beyond this point. */
    PyGILState_Release(gstate);
}

void SignalHandle(const char *data, int size) {
    std::ofstream fs("glog_dump.log", std::ios::app);
    std::string str = std::string(data, size);
    fs << str;
    fs.close();
    LOG(ERROR) << str;
}

TEST(python_module, decode) {
//    google::InitGoogleLogging("main");
//    google::SetStderrLogging(google::INFO);
//    google::InstallFailureSignalHandler();
//    google::InstallFailureWriter(&SignalHandle);
    setenv("PYTHONPATH",
           "../../../3rd_party/pyav/:.:../../python_modules:../../python_module_sdk",
           1);
    time_t time1 = clock();
    Py_Initialize();
    std::string decoder_name;
    std::string decoder_option;
    std::string encoder_name;
    std::string encoder_option;
    JsonParam graph_config;
    graph_config.load("../files/decode_graph.json");
    std::vector<JsonParam> node_jsons;
    graph_config.get_object_list("nodes", node_jsons);
    std::shared_ptr<PythonModule> decoder_module;
//    std::shared_ptr<PythonModule> encode_module;
    get_python_module_info(node_jsons[0], decoder_name, decoder_option);
    std::cout << "create_python_module_from_string start" << std::endl;
    create_python_module_from_string(decoder_name, decoder_option, decoder_module);
    std::cout << "create_python_module_from_string end" << std::endl;
    bool decoder_finished = false;
    bool encoder_finished = false;


    while (true) {
        std::vector<int> input_labels;
        std::vector<int> output_labels;
        output_labels.push_back(0);
        output_labels.push_back(1);
        Task task = Task(0, input_labels, output_labels);
        if (not decoder_finished) {
            decoder_module->process(task);
        }
        if (task.get_timestamp() == DONE) {
            decoder_finished = true;
        }
        input_labels.clear();
        output_labels.clear();
        input_labels.push_back(0);
        input_labels.push_back(1);
        Task encode_task = Task(0, input_labels, output_labels);
        if (decoder_finished)
            break;
    }
    time_t time2 = clock();
    LOG(INFO) << "time:" << time2 - time1 << std::endl;

}

TEST(python_module, decode_encode) {
//        google::InitGoogleLogging("main");
//    google::SetStderrLogging(google::INFO);
//    google::InstallFailureSignalHandler();
//    google::InstallFailureWriter(&SignalHandle);
    setenv("PYTHONPATH",
           "../../../3rd_party/pyav/:.:../../python_modules:../../python_module_sdk",
           1);
    Py_Initialize();
    std::string decoder_name;
    std::string decoder_option;
    std::string encoder_name;
    std::string encoder_option;
    JsonParam graph_config;
    graph_config.load("../files/graph.json");
    std::vector<JsonParam> node_jsons;
    graph_config.get_object_list("nodes", node_jsons);
    std::shared_ptr<PythonModule> decoder_module;
    std::shared_ptr<PythonModule> encode_module;
    get_python_module_info(node_jsons[0], decoder_name, decoder_option);
    get_python_module_info(node_jsons[1], encoder_name, encoder_option);
    create_python_module_from_string(decoder_name, decoder_option, decoder_module);
//    create_python_module(node_jsons[0],decoder_module);
    create_python_module_from_string(encoder_name, encoder_option, encode_module);
//    create_python_module(node_jsons[1],encode_module);
    bool decoder_finished = false;
    bool encoder_finished = false;
//    return ;


    while (true) {
        std::vector<int> input_labels;
        std::vector<int> output_labels;
        output_labels.push_back(0);
        output_labels.push_back(1);
        Task task = Task(0, input_labels, output_labels);
        if (not decoder_finished) {
            decoder_module->process(task);
        }
        if (task.get_timestamp() == DONE) {
            decoder_finished = true;
        }
        input_labels.clear();
        output_labels.clear();
        input_labels.push_back(0);
        input_labels.push_back(1);
        Task encode_task = Task(0, input_labels, output_labels);
        Packet packet;
        bool has_data = false;
        while (task.pop_packet_from_out_queue(0, packet)) {
            has_data = true;
            encode_task.fill_input_packet(0, packet);
        }
        while (task.pop_packet_from_out_queue(1, packet)) {
            has_data = true;
            encode_task.fill_input_packet(1, packet);
        }
        if (has_data)
            encode_module->process(encode_task);
        if (decoder_finished)
            break;
    }
}

TEST(python_module, gil) {
    setenv("PYTHONPATH",
           "../../../3rd_party/pyav/:.:../../python_modules:../../python_module_sdk",
           1);
    Py_Initialize();
    PyEval_InitThreads();
//    PyEval_ReleaseThread(PyThreadState_Get());
    PyThreadState *state = PyEval_SaveThread();
    PyGILState_STATE new_state = PyGILState_Ensure();
    boost::python::object python_file = boost::python::import("timestamp");
    boost::python::list output_stream;
    boost::python::list input_stream;
    /* Release the thread. No Python API allowed beyond this point. */
    PyGILState_Release(new_state);
    PyGILState_Ensure();

}

TEST(python_module, decode_filter_encoder) {
    setenv("PYTHONPATH",
           "../../../3rd_party/pyav/:.:../../python_modules:../../python_module_sdk",
           1);
    Py_Initialize();
    std::string decoder_name;
    std::string decoder_option;
    std::string filter_name;
    std::string filter_opiton;
    std::string encoder_name;
    std::string encoder_option;
    JsonParam graph_config;
    graph_config.load("../files/filter_opt_graph.json");
    std::vector<JsonParam> node_jsons;
    graph_config.get_object_list("nodes", node_jsons);
    std::shared_ptr<PythonModule> decoder_module;
    std::shared_ptr<PythonModule> filter_module;
    std::shared_ptr<PythonModule> encode_module;
    get_python_module_info(node_jsons[0], decoder_name, decoder_option);
    get_python_module_info(node_jsons[1], filter_name, filter_opiton);
    get_python_module_info(node_jsons[2], encoder_name, encoder_option);
    create_python_module_from_string(decoder_name, decoder_option, decoder_module);

//    create_python_module(node_jsons[0],decoder_module);
//    filter_opiton= "{\"filters\": [{\"inputs\": [{\"stream\": 0,\"pin\": 0}],\"name\": \"scale\",\"para\": \"100:200\",\"outputs\": [   {\"stream\": 0,\"pin\": 0}]}]}";
    create_python_module_from_string(filter_name, filter_opiton, filter_module);

    create_python_module_from_string(encoder_name, encoder_option, encode_module);
//    create_python_module(node_jsons[1],encode_module);
    bool decoder_finished = false;
    bool encoder_finished = false;
//    return ;


    while (true) {
        std::vector<int> input_labels;
        std::vector<int> output_labels;
        output_labels.push_back(0);
        output_labels.push_back(1);
        Task task = Task(0, input_labels, output_labels);
        if (not decoder_finished) {
            decoder_module->process(task);
        }
        if (task.get_timestamp() == DONE) {
            decoder_finished = true;
        }
        input_labels.clear();
        output_labels.clear();
        input_labels.push_back(0);
        output_labels.push_back(0);
        Task filter_task = Task(0, input_labels, output_labels);
        Packet packet;
        bool has_data = false;
        while (task.pop_packet_from_out_queue(0, packet)) {
            has_data = true;
            filter_task.fill_input_packet(0, packet);
        }

        if (has_data) {
            filter_module->process(filter_task);
        }

        input_labels.clear();
        output_labels.clear();
        input_labels.push_back(0);
        input_labels.push_back(1);
        Task encode_task = Task(0, input_labels, output_labels);

        has_data = false;

        while (task.pop_packet_from_out_queue(1, packet)) {
            has_data = true;
            encode_task.fill_input_packet(1, packet);
        }
        while (filter_task.pop_packet_from_out_queue(0, packet)) {
            has_data = true;
            encode_task.fill_input_packet(0, packet);
        }
        if (has_data)
            encode_module->process(encode_task);
        if (decoder_finished)
            break;
    }
}
