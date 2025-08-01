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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <utility>

#include "../include/builder.hpp"
#include "../../c_engine/include/module_factory.h"
#include "../../c_engine/include/graph_config.h"
#include "../../c_engine/include/optimizer.h"
#include "../../c_engine/include/common.h"
#include "../include/connector.hpp"

namespace bmf::builder {
namespace internal {
RealStream::RealStream(const std::shared_ptr<RealNode> &node, std::string name,
                       std::string notify, std::string alias)
    : node_(node), graph_(node->graph_), name_(std::move(name)),
      notify_(std::move(notify)), alias_(std::move(alias)) {}

RealStream::RealStream(const std::shared_ptr<RealGraph> &graph,
                       std::string name, std::string notify, std::string alias)
    : graph_(graph), name_(std::move(name)), notify_(std::move(notify)),
      alias_(std::move(alias)) {
    std::shared_ptr<RealNode> nil;
    node_ = nil;
}

void RealStream::SetNotify(std::string const &notify) {
    auto node = node_.lock();
    if (!node)
        throw std::logic_error("Could not call SetNotify on an input stream.");
    int idx = -1;
    for (idx = 0; idx < node->outputStreams_.size(); ++idx)
        if (node->outputStreams_[idx]->name_ == name_)
            break;
    if (idx < 0)
        throw std::logic_error("Internal error.");
    node->GiveStreamNotify(idx, notify);
}

void RealStream::SetAlias(std::string const &alias) {
    auto node = node_.lock();
    if (!node)
        throw std::logic_error("Could not call SetAlias on an input stream.");
    int idx = -1;
    for (idx = 0; idx < node->outputStreams_.size(); ++idx)
        if (node->outputStreams_[idx]->name_ == name_)
            break;
    if (idx < 0)
        throw std::logic_error("Internal error.");
    node->GiveStreamAlias(idx, alias);
}

std::shared_ptr<RealNode> RealStream::AddModule(
    std::string const &alias, const bmf_sdk::JsonParam &option,
    std::vector<std::shared_ptr<RealStream>> inputStreams,
    std::string const &moduleName, ModuleType moduleType,
    std::string const &modulePath, std::string const &moduleEntry,
    InputManagerType inputStreamManager, int scheduler) {
    inputStreams.insert(inputStreams.begin(), shared_from_this());
    return graph_.lock()->AddModule(alias, option, inputStreams, moduleName,
                                    moduleType, modulePath, moduleEntry,
                                    inputStreamManager, scheduler);
}

nlohmann::json RealStream::Dump() {
    nlohmann::json info;

    info["identifier"] = (notify_.empty() ? "" : (notify_ + ":")) + name_;
    info["alias"] = alias_;

    return info;
}

void RealStream::Start() {
    std::vector<std::shared_ptr<internal::RealStream>> generateRealStreams;
    generateRealStreams.emplace_back(shared_from_this());
    graph_.lock()->Start(generateRealStreams, false, true);
}

std::string RealStream::GetName() { return name_; }

RealNode::ModuleMetaInfo::ModuleMetaInfo(std::string moduleName,
                                         ModuleType moduleType,
                                         std::string modulePath,
                                         std::string moduleEntry)
    : moduleName_(std::move(moduleName)), moduleType_(moduleType),
      modulePath_(std::move(modulePath)), moduleEntry_(std::move(moduleEntry)) {
}

nlohmann::json RealNode::ModuleMetaInfo::Dump() {
    nlohmann::json info;

    switch (moduleType_) {
    case C:
        info["type"] = "c";
        break;
    case CPP:
        info["type"] = "c++";
        break;
    case Python:
        info["type"] = "python";
        break;
    case Go:
        info["type"] = "go";
        break;
    }
    info["name"] = moduleName_;
    info["path"] = modulePath_;
    info["entry"] = moduleEntry_;

    return info;
}

nlohmann::json RealNode::NodeMetaInfo::Dump() {
    nlohmann::json info;

    info["premodule_id"] = preModuleUID_;
    info["callback_binding"] = 
        nlohmann::json(std::vector<std::string>());
    for (auto &kv : callbackBinding_) {
        info["callback_binding"].push_back(
            std::to_string(kv.first) + ":" + std::to_string(kv.second)
        );
    }

    return info;
}

RealNode::RealNode(const std::shared_ptr<RealGraph> &graph, int id,
                   std::string alias, const bmf_sdk::JsonParam &option,
                   std::vector<std::shared_ptr<RealStream>> inputStreams,
                   std::string const &moduleName, ModuleType moduleType,
                   std::string const &modulePath,
                   std::string const &moduleEntry,
                   InputManagerType inputStreamManager, int scheduler)
    : graph_(graph), id_(id), alias_(std::move(alias)), option_(option),
      moduleInfo_({moduleName, moduleType, modulePath, moduleEntry}),
      metaInfo_(), inputStreams_(std::move(inputStreams)),
      inputManager_(inputStreamManager), scheduler_(scheduler) {
    //            outputStreams_.reserve(BMF_MAX_CAPACITY);
}

std::shared_ptr<RealStream> RealNode::Stream(int idx) {
    if (idx < 0)
        throw std::overflow_error("Requesting unexisted stream using index.");
    //            if (idx >= BMF_MAX_CAPACITY)
    //                throw std::overflow_error("Stream index bigger than max
    //                capacity (1024 by default).");
    for (auto i = outputStreams_.size(); i <= idx; ++i) {
        auto buf = new char[255];
        std::sprintf(buf, "%s_%d_%lu", moduleInfo_.moduleName_.c_str(), id_, i);
        outputStreams_.emplace_back(std::move(std::make_shared<RealStream>(
            shared_from_this(), std::string(buf), "", "")));
        delete[] buf;
    }
    return outputStreams_[idx];
}

std::shared_ptr<RealStream> RealNode::Stream(std::string const &name) {
    auto graph = graph_.lock();
    if (graph->existedStreamAlias_.count(name) &&
        graph->existedStreamAlias_[name]->node_.lock().get() == this)
        return graph->existedStreamAlias_[name];
    if (existedStreamNotify_.count(name))
        return existedStreamNotify_[name];

    throw std::logic_error(
        "Requesting unexisted stream using name. (Not an alias nor notify.)");
}

void RealNode::SetAlias(std::string const &alias) {
    graph_.lock()->GiveNodeAlias(shared_from_this(), alias);
}

void RealNode::GiveStreamNotify(int idx, std::string const &notify) {
    auto graph = graph_.lock();
    if (graph->existedNodeAlias_.count(notify))
        throw std::logic_error(
            "Duplicated stream notify with existing node alias.");
    if (graph->existedStreamAlias_.count(notify))
        throw std::logic_error(
            "Duplicated stream notify with existing stream alias.");
    if (existedStreamNotify_.count(notify))
        throw std::logic_error(
            "Duplicated stream notify with existing stream notify.");
    existedStreamNotify_[notify] = outputStreams_[idx];
    outputStreams_[idx]->notify_ = notify;
}

void RealNode::GiveStreamAlias(int idx, std::string const &alias) {
    graph_.lock()->GiveStreamAlias(outputStreams_[idx], alias);
}

void RealNode::SetInputManager(InputManagerType inputStreamManager) {
    if (graph_.lock()->mode_ == ServerMode) {
        if (inputStreamManager != Server)
            throw std::logic_error("cannot set input_manager other than Server "
                                   "to node in graph set to ServerMode");
    }
    inputManager_ = inputStreamManager;
}

void RealNode::SetScheduler(int scheduler) { scheduler_ = scheduler; }

void RealNode::SetPreModule(bmf::BMFModule preModuleInstance) {
    metaInfo_.preModuleInstance_ =
        std::make_shared<bmf::BMFModule>(preModuleInstance);
    metaInfo_.preModuleUID_ = preModuleInstance.uid();
}

void RealNode::AddCallback(long long key, bmf::BMFCallback callbackInstance) {
    metaInfo_.callbackInstances_[key] =
        std::make_shared<bmf::BMFCallback>(callbackInstance);
    metaInfo_.callbackBinding_[key] = callbackInstance.uid();
}
void RealNode::AddCallback(long long key, std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> callback) {
    auto cb = bmf::BMFCallback(callback);
    metaInfo_.callbackInstances_[key] =
        std::make_shared<bmf::BMFCallback>(cb);
    metaInfo_.callbackBinding_[key] = cb.uid();
}

std::shared_ptr<RealNode>
RealNode::AddModule(std::string const &alias, const bmf_sdk::JsonParam &option,
                    std::vector<std::shared_ptr<RealStream>> inputStreams,
                    std::string const &moduleName, ModuleType moduleType,
                    std::string const &modulePath,
                    std::string const &moduleEntry,
                    InputManagerType inputStreamManager, int scheduler) {
    inputStreams.insert(inputStreams.begin(), Stream(0));
    return graph_.lock()->AddModule(alias, option, inputStreams, moduleName,
                                    moduleType, modulePath, moduleEntry,
                                    inputStreamManager, scheduler);
}

nlohmann::json RealNode::Dump() {
    nlohmann::json info;

    info["id"] = id_;
    info["alias"] = alias_;
    info["module_info"] = moduleInfo_.Dump();
    info["meta_info"] = metaInfo_.Dump();
    info["input_streams"] = nlohmann::json::array();
    for (auto &s : inputStreams_)
        info["input_streams"].push_back(s->Dump());
    info["output_streams"] = nlohmann::json::array();
    for (auto &s : outputStreams_)
        info["output_streams"].push_back(s->Dump());
    info["option"] = option_.json_value_;
    info["scheduler"] = scheduler_;
    switch (inputManager_) {
    case Default:
        info["input_manager"] = "default";
        break;
    case Immediate:
        info["input_manager"] = "immediate";
        break;
    case Server:
        info["input_manager"] = "server";
        break;
    case FrameSync:
        info["input_manager"] = "framesync";
        break;
    case ClockSync:
        info["input_manager"] = "clocksync";
        break;
    default:
        info["input_manager"] = "default";
    }

    return info;
}

RealGraph::RealGraph(GraphMode runMode, const bmf_sdk::JsonParam &graphOption)
    : mode_(runMode), graphOption_(graphOption), placeholderNode_(nullptr) {}

void RealGraph::GiveStreamAlias(std::shared_ptr<RealStream> stream,
                                std::string const &alias) {
    if (existedNodeAlias_.count(alias))
        throw std::logic_error(
            "Duplicated stream alias with existing node alias.");
    if (existedStreamAlias_.count(alias))
        throw std::logic_error(
            "Duplicated stream alias with existing stream alias.");
    for (auto &nd : nodes_)
        if (nd->existedStreamNotify_.count(alias))
            throw std::logic_error(
                "Duplicated stream alias with existing stream notify.");
    existedStreamAlias_[alias] = std::move(stream);
    existedStreamAlias_[alias]->alias_ = alias;
}

void RealGraph::GiveNodeAlias(std::shared_ptr<RealNode> node,
                              std::string const &alias) {
    if (existedNodeAlias_.count(alias))
        throw std::logic_error(
            "Duplicated node alias with existing node alias.");
    if (existedStreamAlias_.count(alias))
        throw std::logic_error(
            "Duplicated node alias with existing stream alias.");
    for (auto &nd : nodes_)
        if (nd->existedStreamNotify_.count(alias))
            throw std::logic_error(
                "Duplicated node alias with existing stream notify.");
    existedNodeAlias_[alias] = std::move(node);
    existedNodeAlias_[alias]->alias_ = alias;
}

std::shared_ptr<RealNode> RealGraph::AddModule(
    std::string const &alias, const bmf_sdk::JsonParam &option,
    const std::vector<std::shared_ptr<RealStream>> &inputStreams,
    std::string const &moduleName, ModuleType moduleType,
    std::string const &modulePath, std::string const &moduleEntry,
    InputManagerType inputStreamManager, int scheduler) {
    //            if (nodes_.size() + 1 >= BMF_MAX_CAPACITY)
    //                throw std::overflow_error("Node number bigger than max
    //                capacity (1024 by default).");
    if (mode_ == ServerMode)
        inputStreamManager = Server;
    int node_id = nodes_.size();
    nodes_.emplace_back(std::move(std::make_shared<RealNode>(
        shared_from_this(), node_id, alias, option, inputStreams, moduleName,
        moduleType, modulePath, moduleEntry, inputStreamManager, scheduler)));
    return nodes_[node_id];
}

std::shared_ptr<RealNode> RealGraph::GetAliasedNode(std::string const &alias) {
    if (!existedNodeAlias_.count(alias))
        throw std::logic_error("Unexisted aliased node.");
    return existedNodeAlias_[alias];
}

std::shared_ptr<RealStream>
RealGraph::GetAliasedStream(std::string const &alias) {
    if (!existedStreamAlias_.count(alias))
        throw std::logic_error("Unexisted aliased stream.");
    return existedStreamAlias_[alias];
}

std::shared_ptr<RealStream> RealGraph::NewPlaceholderStream() {
    if (placeholderNode_ == nullptr)
        placeholderNode_ = std::move(std::make_shared<RealNode>(
            shared_from_this(), std::numeric_limits<int>::max(), "",
            bmf_sdk::JsonParam(), std::vector<std::shared_ptr<RealStream>>(),
            "BMFPlaceholderNode", CPP, "", "", Immediate, 0));

    return placeholderNode_->Stream(placeholderNode_->outputStreams_.size());
}

nlohmann::json RealGraph::Dump() {
    nlohmann::json info;

    info["input_streams"] = nlohmann::json::array();
    info["output_streams"] = nlohmann::json::array();
    info["nodes"] = nlohmann::json::array();
    info["option"] = graphOption_.json_value_;
    switch (mode_) {
    case NormalMode:
        info["mode"] = "Normal";
        break;
    case ServerMode:
        info["mode"] = "Server";
        break;
    case GeneratorMode:
        info["mode"] = "Generator";
        break;
    case SubGraphMode:
        info["mode"] = "Subgraph";
        break;
    case PushDataMode:
        info["mode"] = "Pushdata";
        break;
    }
    for (auto &nd : nodes_)
        info["nodes"].push_back(nd->Dump());
    for (auto &s : inputStreams_)
        info["input_streams"].push_back(s->Dump());
    for (auto &s : outputStreams_)
        info["output_streams"].push_back(s->Dump());

    return info;
}

void RealGraph::SetOption(const bmf_sdk::JsonParam &optionPatch) {
    graphOption_.merge_patch(optionPatch);
}

void RealGraph::Start(bool dumpGraph, bool needMerge) {
    auto graph_config = Dump().dump(4);
    BMFLOG(BMF_INFO) << graph_config << std::endl;
    if (dumpGraph || (graphOption_.json_value_.count("dump_graph") &&
                      graphOption_.json_value_["dump_graph"] == 1)) {
        BMFLOG(BMF_INFO) << "graph_config dump" << std::endl;
        std::ofstream graph_file("graph.json", std::ios::app);
        BMFLOG(BMF_INFO) << graph_config << std::endl;
        graph_file << graph_config;
        graph_file.close();
    }
    if (graphInstance_ == nullptr)
        graphInstance_ =
            std::make_shared<bmf::BMFGraph>(graph_config, false, needMerge);
    graphInstance_->start();
}

int RealGraph::Run(bool dumpGraph, bool needMerge) {
    auto graph_config = Dump().dump(4);
    if (dumpGraph || (graphOption_.json_value_.count("dump_graph") &&
                      graphOption_.json_value_["dump_graph"] == 1)) {
        std::ofstream graph_file("graph.json", std::ios::app);
        graph_file << graph_config;
        graph_file.close();
    }
    if (graphInstance_ == nullptr)
        graphInstance_ =
            std::make_shared<bmf::BMFGraph>(graph_config, false, needMerge);
    graphInstance_->start();
    return graphInstance_->close();
}

void RealGraph::Start(
    const std::vector<std::shared_ptr<internal::RealStream>> &streams,
    bool dumpGraph, bool needMerge) {
    outputStreams_.insert(outputStreams_.end(), streams.begin(), streams.end());
    Start(dumpGraph, needMerge);
}

int RealGraph::Close() {
    return graphInstance_->close();
}

int RealGraph::ForceClose() {
    return graphInstance_->force_close();
}

bmf::BMFGraph RealGraph::Instantiate(bool dumpGraph, bool needMerge) {
    auto graph_config = Dump().dump(4);
    if (dumpGraph || (graphOption_.json_value_.count("dump_graph") &&
                      graphOption_.json_value_["dump_graph"] == 1)) {
        std::ofstream graph_file("graph.json", std::ios::app);
        graph_file << graph_config;
        graph_file.close();
    }
    if (graphInstance_ == nullptr)
        graphInstance_ =
            std::make_shared<bmf::BMFGraph>(graph_config, false, needMerge);
    return *graphInstance_;
}

bmf::BMFGraph RealGraph::Instance() {
    if (graphInstance_ == nullptr)
        throw std::logic_error(
            "trying to get graph instance before instantiated.");
    return *graphInstance_;
}

Packet RealGraph::Generate(std::string streamName, bool block) {
    return graphInstance_->poll_output_stream_packet(streamName, block);
}

int RealGraph::FillPacket(std::string streamName, Packet packet, bool block) {
    return graphInstance_->add_input_stream_packet(streamName, packet, block);
}

std::shared_ptr<RealStream> RealGraph::InputStream(std::string streamName,
                                                   std::string notify,
                                                   std::string alias) {
    auto realStream = std::make_shared<internal::RealStream>(
        shared_from_this(), streamName, notify, alias);
    inputStreams_.emplace_back(realStream);
    return realStream;
}

} // namespace internal

std::string GetVersion() { return BMF_VERSION; }

std::string GetCommit() { return BMF_COMMIT; }

void ChangeDmpPath(std::string path) { bmf::ChangeDmpPath(path); }

bmf::BMFModule GetModuleInstance(std::string const &moduleName,
                                 std::string const &option,
                                 ModuleType moduleType,
                                 std::string const &modulePath,
                                 std::string const &moduleEntry) {
    std::string type_;
    switch (moduleType) {
    case C:
        type_ = "c";
        break;
    case CPP:
        type_ = "c++";
        break;
    case Python:
        type_ = "python";
        break;
    case Go:
        type_ = "go";
        break;
    }
    return bmf::BMFModule(moduleName, option, type_, modulePath, moduleEntry);
}

bmf::BMFCallback
GetCallbackInstance(std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> callback) {
    return bmf::BMFCallback(std::move(callback));
}

Stream::Stream(std::shared_ptr<internal::RealStream> baseP)
    : baseP_(std::move(baseP)) {}

void Stream::SetNotify(std::string const &notify) { baseP_->SetNotify(notify); }

void Stream::SetAlias(std::string const &alias) { baseP_->SetAlias(alias); }

void Stream::Start() { baseP_->Start(); }

Node Stream::Module(const std::vector<Stream> &inStreams,
                    std::string const &moduleName, ModuleType moduleType,
                    const bmf_sdk::JsonParam &option, std::string const &alias,
                    std::string const &modulePath,
                    std::string const &moduleEntry,
                    InputManagerType inputStreamManager, int scheduler) {
    return ConnectNewModule(alias, option, inStreams, moduleName, moduleType,
                            modulePath, moduleEntry, inputStreamManager,
                            scheduler);
}

Node Stream::CppModule(const std::vector<Stream> &inStreams,
                       std::string const &moduleName,
                       const bmf_sdk::JsonParam &option,
                       std::string const &alias, std::string const &modulePath,
                       std::string const &moduleEntry,
                       InputManagerType inputStreamManager, int scheduler) {
    return ConnectNewModule(alias, option, inStreams, moduleName, CPP,
                            modulePath, moduleEntry, inputStreamManager,
                            scheduler);
}

Node Stream::PythonModule(const std::vector<Stream> &inStreams,
                          std::string const &moduleName,
                          const bmf_sdk::JsonParam &option,
                          std::string const &alias,
                          std::string const &modulePath,
                          std::string const &moduleEntry,
                          InputManagerType inputStreamManager, int scheduler) {
    return ConnectNewModule(alias, option, inStreams, moduleName, Python,
                            modulePath, moduleEntry, inputStreamManager,
                            scheduler);
}

Node Stream::GoModule(const std::vector<Stream> &inStreams,
                      std::string const &moduleName,
                      const bmf_sdk::JsonParam &option,
                      std::string const &alias, std::string const &modulePath,
                      std::string const &moduleEntry,
                      InputManagerType inputStreamManager, int scheduler) {
    return ConnectNewModule(alias, option, inStreams, moduleName, Go,
                            modulePath, moduleEntry, inputStreamManager,
                            scheduler);
}

Node Stream::Decode(const bmf_sdk::JsonParam &decodePara,
                    std::string const &alias) {
    auto nd = ConnectNewModule(alias, decodePara, {}, "c_ffmpeg_decoder", CPP,
                               "", "", Immediate, 0);
    nd[0].SetNotify("video");
    nd[1].SetNotify("audio");
    return nd;
}

Node Stream::EncodeAsVideo(const bmf_sdk::JsonParam &encodePara,
                           std::string const &alias) {
    return ConnectNewModule(alias, encodePara, {}, "c_ffmpeg_encoder", CPP, "",
                            "", Immediate, 1);
}

Node Stream::EncodeAsVideo(Stream audioStream,
                           const bmf_sdk::JsonParam &encodePara,
                           std::string const &alias) {
    return ConnectNewModule(alias, encodePara, {std::move(audioStream)},
                            "c_ffmpeg_encoder", CPP, "", "", Immediate, 1);
}

Node Stream::FFMpegFilter(const std::vector<Stream> &inStreams,
                          std::string const &filterName,
                          bmf_sdk::JsonParam filterPara,
                          std::string const &alias) {
    nlohmann::json realPara;
    realPara["name"] = filterName;
    realPara["para"] = filterPara.json_value_;
    filterPara = bmf_sdk::JsonParam(realPara);
    return ConnectNewModule(alias, filterPara, inStreams, "c_ffmpeg_filter",
                            CPP, "", "", Immediate, 0);
}

Node Stream::Fps(int fps, std::string const &alias) {
    bmf_sdk::JsonParam para;
    para.json_value_["fps"] = fps;
    return FFMpegFilter({}, "fps", para, alias);
}

Node Stream::InternalFFMpegFilter(const std::vector<Stream> &inStreams,
                                  std::string const &filterName,
                                  const bmf_sdk::JsonParam &filterPara,
                                  std::string const &alias) {
    return ConnectNewModule(alias, filterPara, inStreams, "c_ffmpeg_filter",
                            CPP, "", "", Immediate, 0);
}

Node Stream::ConnectNewModule(
    const std::string &alias, const bmf_sdk::JsonParam &option,
    const std::vector<Stream> &inputStreams, const std::string &moduleName,
    ModuleType moduleType, const std::string &modulePath,
    const std::string &moduleEntry, InputManagerType inputStreamManager,
    int scheduler) {
    std::vector<std::shared_ptr<internal::RealStream>> inRealStreams;
    inRealStreams.reserve(inputStreams.size());
    for (auto &s : inputStreams)
        inRealStreams.emplace_back(s.baseP_);
    return Node(baseP_->AddModule(alias, option, inRealStreams, moduleName,
                                  moduleType, modulePath, moduleEntry,
                                  inputStreamManager, scheduler));
}

std::string Stream::GetName() { return baseP_->GetName(); }

Node::Node(std::shared_ptr<internal::RealNode> baseP)
    : baseP_(std::move(baseP)) {}

class Stream Node::operator[](int index) { return Stream(index); }

class Stream Node::operator[](std::string const &notifyOrAlias) {
    return Stream(notifyOrAlias);
}

class Stream Node::Stream(int index) {
    return (class Stream)(baseP_->Stream(index));
}

class Stream Node::Stream(std::string const &notifyOrAlias) {
    return (class Stream)(baseP_->Stream(notifyOrAlias));
}

Node::operator class Stream() { return Stream(0); }

void Node::SetAlias(std::string const &alias) { baseP_->SetAlias(alias); }

void Node::SetInputStreamManager(InputManagerType inputStreamManager) {
    baseP_->SetInputManager(inputStreamManager);
}

void Node::SetThread(int threadNum) { baseP_->SetScheduler(threadNum); }

void Node::SetPreModule(const bmf::BMFModule &preModuleInstance) {
    baseP_->SetPreModule(preModuleInstance);
}

void Node::AddCallback(long long key,
                       const bmf::BMFCallback &callbackInstance) {
    baseP_->AddCallback(key, callbackInstance);
}

void Node::AddCallback(long long key, 
                       std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> callback) {
    baseP_->AddCallback(key, callback);
}

void Node::Start() { Stream(0).Start(); }

Node Node::Module(const std::vector<class Stream> &inStreams,
                  std::string const &moduleName, ModuleType moduleType,
                  const bmf_sdk::JsonParam &option, std::string const &alias,
                  std::string const &modulePath, std::string const &moduleEntry,
                  InputManagerType inputStreamManager, int scheduler) {
    return ConnectNewModule(alias, option, inStreams, moduleName, moduleType,
                            modulePath, moduleEntry, inputStreamManager,
                            scheduler);
}

Node Node::CppModule(const std::vector<class Stream> &inStreams,
                     std::string const &moduleName,
                     const bmf_sdk::JsonParam &option, std::string const &alias,
                     std::string const &modulePath,
                     std::string const &moduleEntry,
                     InputManagerType inputStreamManager, int scheduler) {
    return ConnectNewModule(alias, option, inStreams, moduleName, CPP,
                            modulePath, moduleEntry, inputStreamManager,
                            scheduler);
}

Node Node::PythonModule(const std::vector<class Stream> &inStreams,
                        std::string const &moduleName,
                        const bmf_sdk::JsonParam &option,
                        std::string const &alias, std::string const &modulePath,
                        std::string const &moduleEntry,
                        InputManagerType inputStreamManager, int scheduler) {
    return ConnectNewModule(alias, option, inStreams, moduleName, Python,
                            modulePath, moduleEntry, inputStreamManager,
                            scheduler);
}

Node Node::GoModule(const std::vector<class Stream> &inStreams,
                    std::string const &moduleName,
                    const bmf_sdk::JsonParam &option, std::string const &alias,
                    std::string const &modulePath,
                    std::string const &moduleEntry,
                    InputManagerType inputStreamManager, int scheduler) {
    return ConnectNewModule(alias, option, inStreams, moduleName, Go,
                            modulePath, moduleEntry, inputStreamManager,
                            scheduler);
}

Node Node::Decode(const bmf_sdk::JsonParam &decodePara,
                  std::string const &alias) {
    auto nd = ConnectNewModule(alias, decodePara, {}, "c_ffmpeg_decoder", CPP,
                               "", "", Immediate, 0);
    nd[0].SetNotify("video");
    nd[1].SetNotify("audio");
    return nd;
}

Node Node::EncodeAsVideo(const bmf_sdk::JsonParam &encodePara,
                         std::string const &alias) {
    return ConnectNewModule(alias, encodePara, {}, "c_ffmpeg_encoder", CPP, "",
                            "", Immediate, 1);
}

Node Node::EncodeAsVideo(class Stream audioStream,
                         const bmf_sdk::JsonParam &encodePara,
                         std::string const &alias) {
    return ConnectNewModule(alias, encodePara, {std::move(audioStream)},
                            "c_ffmpeg_encoder", CPP, "", "", Immediate, 1);
}

Node Node::FFMpegFilter(const std::vector<class Stream> &inStreams,
                        std::string const &filterName,
                        bmf_sdk::JsonParam filterPara,
                        std::string const &alias) {
    nlohmann::json realPara;
    realPara["name"] = filterName;
    realPara["para"] = filterPara.json_value_;
    filterPara = bmf_sdk::JsonParam(realPara);
    return ConnectNewModule(alias, filterPara, inStreams, "c_ffmpeg_filter",
                            CPP, "", "", Immediate, 0);
}

Node Node::Fps(int fps, std::string const &alias) {
    bmf_sdk::JsonParam para;
    para.json_value_["fps"] = fps;
    return FFMpegFilter({}, "fps", para, alias);
}

Node Node::InternalFFMpegFilter(const std::vector<class Stream> &inStreams,
                                std::string const &filterName,
                                const bmf_sdk::JsonParam &filterPara,
                                std::string const &alias) {
    return ConnectNewModule(alias, filterPara, inStreams, "c_ffmpeg_filter",
                            CPP, "", "", Immediate, 0);
}

Node Node::ConnectNewModule(
    std::string const &alias, const bmf_sdk::JsonParam &option,
    const std::vector<class Stream> &inputStreams,
    std::string const &moduleName, ModuleType moduleType,
    std::string const &modulePath, std::string const &moduleEntry,
    InputManagerType inputStreamManager, int scheduler) {
    std::vector<std::shared_ptr<internal::RealStream>> inRealStreams;
    inRealStreams.reserve(inputStreams.size());
    for (auto &s : inputStreams)
        inRealStreams.emplace_back(s.baseP_);
    return Node(baseP_->AddModule(alias, option, inRealStreams, moduleName,
                                  moduleType, modulePath, moduleEntry,
                                  inputStreamManager, scheduler));
}

Graph::Graph(GraphMode runMode, bmf_sdk::JsonParam graphOption)
    : graph_(std::make_shared<internal::RealGraph>(runMode, graphOption)) {}

Graph::Graph(GraphMode runMode, nlohmann::json graphOption)
    : graph_(std::make_shared<internal::RealGraph>(
          runMode, bmf_sdk::JsonParam(graphOption))) {}

bmf::BMFGraph Graph::Instantiate(bool dumpGraph, bool needMerge) {
    return graph_->Instantiate(dumpGraph, needMerge);
}

bmf::BMFGraph Graph::Instance() { return graph_->Instance(); }

int Graph::Run(bool dumpGraph, bool needMerge) {
    return graph_->Run(dumpGraph, needMerge);
}

void Graph::Start(bool dumpGraph, bool needMerge) {
    graph_->Start(dumpGraph, needMerge);
}

void Graph::Start(std::vector<Stream> &generateStreams, bool dumpGraph,
                  bool needMerge) {
    std::vector<std::shared_ptr<internal::RealStream>> generateRealStreams;
    generateRealStreams.reserve(generateStreams.size());
    for (auto &s : generateStreams)
        generateRealStreams.emplace_back(s.baseP_);
    graph_->Start(generateRealStreams, dumpGraph, needMerge);
}

int Graph::Close() {
    return graph_->Close();
}

int Graph::ForceClose() {
    return graph_->ForceClose();
}

Packet Graph::Generate(std::string streamName, bool block) {
    return graph_->Generate(streamName, block);
}

void Graph::SetTotalThreadNum(int num) {
    graph_->graphOption_.json_value_["scheduler_count"] = num;
}

Stream Graph::NewPlaceholderStream() {
    return Stream(graph_->NewPlaceholderStream());
}

Node Graph::GetAliasedNode(std::string const &alias) {
    return Node(graph_->GetAliasedNode(alias));
}

Stream Graph::GetAliasedStream(std::string const &alias) {
    return Stream(graph_->GetAliasedStream(alias));
}

std::string Graph::Dump() { return graph_->Dump().dump(4); }

Node Graph::Module(const std::vector<Stream> &inStreams,
                   std::string const &moduleName, ModuleType moduleType,
                   const bmf_sdk::JsonParam &option, std::string const &alias,
                   std::string const &modulePath,
                   std::string const &moduleEntry,
                   InputManagerType inputStreamManager, int scheduler) {
    return NewNode(alias, option, inStreams, moduleName, moduleType, modulePath,
                   moduleEntry, inputStreamManager, scheduler);
}

Node Graph::CppModule(const std::vector<Stream> &inStreams,
                      std::string const &moduleName,
                      const bmf_sdk::JsonParam &option,
                      std::string const &alias, std::string const &modulePath,
                      std::string const &moduleEntry,
                      InputManagerType inputStreamManager, int scheduler) {
    return NewNode(alias, option, inStreams, moduleName, CPP, modulePath,
                   moduleEntry, inputStreamManager, scheduler);
}

Node Graph::PythonModule(const std::vector<Stream> &inStreams,
                         std::string const &moduleName,
                         const bmf_sdk::JsonParam &option,
                         std::string const &alias,
                         std::string const &modulePath,
                         std::string const &moduleEntry,
                         InputManagerType inputStreamManager, int scheduler) {
    return NewNode(alias, option, inStreams, moduleName, Python, modulePath,
                   moduleEntry, inputStreamManager, scheduler);
}

Node Graph::GoModule(const std::vector<Stream> &inStreams,
                     std::string const &moduleName,
                     const bmf_sdk::JsonParam &option, std::string const &alias,
                     std::string const &modulePath,
                     std::string const &moduleEntry,
                     InputManagerType inputStreamManager, int scheduler) {
    return NewNode(alias, option, inStreams, moduleName, Go, modulePath,
                   moduleEntry, inputStreamManager, scheduler);
}

Node Graph::Decode(const bmf_sdk::JsonParam &decodePara,
                   std::string const &alias,
                   int scheduler) {
    auto nd = NewNode(alias, decodePara, {}, "c_ffmpeg_decoder", CPP, "", "",
                      Immediate, scheduler);
    nd[0].SetNotify("video");
    nd[1].SetNotify("audio");
    return nd;
}

Node Graph::Decode(const bmf_sdk::JsonParam &decodePara, Stream controlStream,
                   std::string const &alias,
                   int scheduler) {
    return NewNode(alias, decodePara, {std::move(controlStream)},
                   "c_ffmpeg_decoder", CPP, "", "", Immediate, scheduler);
}

Node Graph::Encode(Stream videoStream, Stream audioStream,
                   const bmf_sdk::JsonParam &encodePara,
                   std::string const &alias,
                   int scheduler) {
    return NewNode(alias, encodePara,
                   {std::move(videoStream), std::move(audioStream)},
                   "c_ffmpeg_encoder", CPP, "", "", Immediate, scheduler);
}

Node Graph::Encode(Stream videoStream, const bmf_sdk::JsonParam &encodePara,
                   std::string const &alias,
                   int scheduler) {
    return NewNode(alias, encodePara, {std::move(videoStream)},
                   "c_ffmpeg_encoder", CPP, "", "", Immediate, scheduler);
}

Node Graph::Encode(const bmf_sdk::JsonParam &encodePara,
                   std::string const &alias,
                   int scheduler) {
    return NewNode(alias, encodePara, {}, "c_ffmpeg_encoder", CPP, "", "",
                   Immediate, scheduler);
}

Node Graph::FFMpegFilter(const std::vector<Stream> &inStreams,
                         std::string const &filterName,
                         const bmf_sdk::JsonParam &filterPara,
                         std::string const &alias) {
    nlohmann::json realPara;
    realPara["name"] = filterName;
    realPara["para"] = filterPara.json_value_;
    return NewNode(alias, bmf_sdk::JsonParam(realPara), inStreams,
                   "c_ffmpeg_filter", CPP, "", "", Immediate, 0);
}

Node Graph::Fps(Stream inStream, int fps, std::string const &alias) {
    bmf_sdk::JsonParam para;
    para.json_value_["fps"] = fps;
    return FFMpegFilter({std::move(inStream)}, "fps", para, alias);
}

Node Graph::InternalFFMpegFilter(const std::vector<Stream> &inStreams,
                                 std::string const &filterName,
                                 const bmf_sdk::JsonParam &filterPara,
                                 std::string const &alias) {
    return NewNode(alias, filterPara, inStreams, "c_ffmpeg_filter", CPP, "", "",
                   Immediate, 0);
}

Node Graph::NewNode(std::string const &alias, const bmf_sdk::JsonParam &option,
                    const std::vector<Stream> &inputStreams,
                    std::string const &moduleName, ModuleType moduleType,
                    std::string const &modulePath,
                    std::string const &moduleEntry,
                    InputManagerType inputStreamManager, int scheduler) {
    std::vector<std::shared_ptr<internal::RealStream>> inRealStreams;
    inRealStreams.reserve(inputStreams.size());
    for (auto &s : inputStreams)
        inRealStreams.emplace_back(s.baseP_);
    return Node(graph_->AddModule(alias, option, inRealStreams, moduleName,
                                  moduleType, modulePath, moduleEntry,
                                  inputStreamManager, scheduler));
}

SyncModule Graph::Sync(const std::vector<int> inStreams,
                       const std::vector<int> outStreams,
                       bmf_sdk::JsonParam moduleOption,
                       std::string const &moduleName, ModuleType moduleType,
                       std::string const &modulePath,
                       std::string const &moduleEntry, std::string const &alias,
                       InputManagerType inputStreamManager, int scheduler) {
    auto sync_m = SyncModule();
    std::string module_type;
    switch (moduleType) {
    case C:
        module_type = "c";
        break;
    case Python:
        module_type = "python";
        break;
    case Go:
        module_type = "go";
        break;
    default:
        module_type = "c++";
    }
    if (moduleName.compare("c_ffmpeg_filter") == 0) {
        nlohmann::json inputOption;
        nlohmann::json outputOption;
        for (auto id : inStreams) {
            nlohmann::json stream = {
                {"identifier", moduleName + std::to_string(id)}};
            inputOption.push_back(stream);
        }
        for (auto id : outStreams) {
            nlohmann::json stream = {
                {"identifier", moduleName + std::to_string(id)}};
            outputOption.push_back(stream);
        }
        nlohmann::json option = {
            {"option", moduleOption.json_value_},
            {"input_streams", inputOption},
            {"output_streams", outputOption},
        };
        auto config = bmf_engine::NodeConfig(option);
        bmf_engine::Optimizer::convert_filter_para(config);
        bmf_engine::Optimizer::replace_stream_name_with_id(config);
        moduleOption = config.get_option();
    }
    bmf_engine::ModuleFactory::create_module(
        moduleName, -1, moduleOption, module_type, modulePath, moduleEntry,
        sync_m.moduleInstance);
    sync_m.inputStreams = inStreams;
    sync_m.outputStreams = outStreams;
    sync_m.moduleInstance->init();
    return sync_m;
}

SyncModule Graph::Sync(const std::vector<int> inStreams,
                       const std::vector<int> outStreams,
                       nlohmann::json moduleOption,
                       std::string const &moduleName, ModuleType moduleType,
                       std::string const &modulePath,
                       std::string const &moduleEntry, std::string const &alias,
                       InputManagerType inputStreamManager, int scheduler) {
    return Sync(inStreams, outStreams, bmf_sdk::JsonParam(moduleOption),
                moduleName, moduleType, modulePath, moduleEntry, alias,
                inputStreamManager, scheduler);
}

std::map<int, std::vector<Packet>>
Graph::Process(SyncModule module,
               std::map<int, std::vector<Packet>> inputPackets) {
    auto task = bmf_sdk::Task(0, module.inputStreams, module.outputStreams);
    for (auto const &pkts : inputPackets) {
        for (auto const &pkt : pkts.second) {
            task.fill_input_packet(pkts.first, pkt);
        }
    }
    module.moduleInstance->process(task);
    std::map<int, std::vector<Packet>> returnMap;
    for (auto id : module.outputStreams) {
        auto it = task.outputs_queue_.find(id);
        if (it == task.outputs_queue_.end())
            continue;
        while (!it->second->empty()) {
            Packet pkt;
            task.pop_packet_from_out_queue(id, pkt);
            returnMap[id].push_back(pkt);
        }
    }
    return returnMap;
}

SyncPackets Graph::Process(SyncModule module, SyncPackets pkts) {
    SyncPackets returnPkts;
    returnPkts.packets = Process(module, pkts.packets);
    return returnPkts;
}

int32_t Graph::Init(SyncModule module) { return module.moduleInstance->init(); }

int32_t Graph::Close(SyncModule module) {
    return module.moduleInstance->close();
}

int32_t Graph::SendEOF(SyncModule module) {
    auto task = bmf_sdk::Task(0, module.inputStreams, module.outputStreams);
    for (auto id : module.inputStreams) {
        task.fill_input_packet(id, Packet::generate_eof_packet());
    }
    return module.moduleInstance->process(task);
}

void Graph::SetOption(const bmf_sdk::JsonParam &optionPatch) {
    graph_->SetOption(optionPatch);
}

Stream Graph::InputStream(std::string streamName, std::string notify,
                          std::string alias) {
    return Stream(graph_->InputStream(streamName, notify, alias));
}

int Graph::FillPacket(std::string streamName, Packet packet, bool block) {
    return graph_->FillPacket(streamName, packet, block);
}

void SyncPackets::Insert(int streamId, std::vector<Packet> frames) {
    packets.insert(std::make_pair(streamId, frames));
}

std::vector<Packet> SyncPackets::operator[](int index) {
    return packets[index];
}

std::map<int, std::vector<Packet>>
SyncModule::ProcessPkts(std::map<int, std::vector<Packet>> inputPackets) {
    auto task = bmf_sdk::Task(0, inputStreams, outputStreams);
    for (auto const &pkts : inputPackets) {
        for (auto const &pkt : pkts.second) {
            task.fill_input_packet(pkts.first, pkt);
        }
    }
    moduleInstance->process(task);
    std::map<int, std::vector<Packet>> returnMap;
    for (auto id : outputStreams) {
        auto it = task.outputs_queue_.find(id);
        if (it == task.outputs_queue_.end())
            continue;
        while (!it->second->empty()) {
            Packet pkt;
            task.pop_packet_from_out_queue(id, pkt);
            returnMap[id].push_back(pkt);
        }
    }
    return returnMap;
}

SyncPackets SyncModule::ProcessPkts(SyncPackets pkts) {
    SyncPackets returnPkts;
    returnPkts.packets = ProcessPkts(pkts.packets);
    return returnPkts;
}

int32_t SyncModule::Process(bmf_sdk::Task task) {
    return moduleInstance->process(task);
}

int32_t SyncModule::SendEOF() {
    auto task = bmf_sdk::Task(0, inputStreams, outputStreams);
    for (auto id : inputStreams) {
        task.fill_input_packet(id, Packet::generate_eof_packet());
    }
    return moduleInstance->process(task);
}

int32_t SyncModule::Init() { return moduleInstance->init(); }

int32_t SyncModule::Close() { return moduleInstance->close(); }

void SyncModule::DynamicReset(const bmf_sdk::JsonParam& opt_reset) {
    moduleInstance->dynamic_reset(opt_reset);
}
} // namespace bmf::builder

