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

#pragma once
#ifndef BMF_ENGINE_BUILDER_HPP
#define BMF_ENGINE_BUILDER_HPP

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <bmf/sdk/json_param.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/module.h>

#include "connector.hpp"
#include "connector_common.h"

#ifndef BMF_MAX_CAPACITY
#define BMF_MAX_CAPACITY (1 << 10)
#endif

namespace bmf::builder {
class BMF_ENGINE_API Graph;

class BMF_ENGINE_API Node;

class BMF_ENGINE_API Stream;

class BMF_ENGINE_API SyncModule;

class BMF_ENGINE_API SyncPackets;

// enum constants
enum GraphMode {
    NormalMode,
    ServerMode,
    GeneratorMode,
    SubGraphMode,
    UpdateMode,
};
enum InputManagerType { Immediate, Default, Server, FrameSync, ClockSync };
enum ModuleType { CPP, C, Python, Go };
namespace internal {
// Forward declaration
class RealGraph;

class RealNode;

class RealStream;

class RealStream : public std::enable_shared_from_this<RealStream> {
  public:
    RealStream(const std::shared_ptr<RealNode> &node, std::string name,
               std::string notify, std::string alias);

    RealStream(const std::shared_ptr<RealGraph>& graph, std::string name, std::string notify,
                       std::string alias);

    RealStream() = delete;

    RealStream(RealStream const &) = delete;

    RealStream(RealStream &&) = default;

    void SetNotify(std::string const &notify);

    void SetAlias(std::string const &alias);
    void Start();

    nlohmann::json Dump();

    std::shared_ptr<RealNode>
    AddModule(std::string const &alias, const bmf_sdk::JsonParam &option,
              std::vector<std::shared_ptr<RealStream>> inputStreams,
              std::string const &moduleName, ModuleType moduleType,
              std::string const &modulePath, std::string const &moduleEntry,
              InputManagerType inputStreamManager, int scheduler);
    std::string GetName();

  private:
    friend bmf::builder::Graph;
    friend bmf::builder::Node;
    friend bmf::builder::Stream;
    friend RealGraph;
    friend RealNode;

    std::weak_ptr<RealNode> node_;
    std::weak_ptr<RealGraph> graph_;
    std::string name_;
    std::string notify_;
    std::string alias_;
};

class RealNode : public std::enable_shared_from_this<RealNode> {
  public:
    RealNode(const std::shared_ptr<RealGraph> &graph, int id, std::string alias,
             const bmf_sdk::JsonParam &option,
             std::vector<std::shared_ptr<RealStream>> inputStreams,
             std::string const &moduleName, ModuleType mduleType,
             std::string const &modulePath, std::string const &moduleEntry,
             InputManagerType inputStreamManager, int scheduler);

    RealNode() = delete;

    RealNode(RealNode const &) = delete;

    RealNode(RealNode &&) = default;

    std::shared_ptr<RealStream> Stream(int idx);

    std::shared_ptr<RealStream> Stream(std::string const &name);

    void GiveStreamNotify(int idx, std::string const &notify);

    void GiveStreamAlias(int idx, std::string const &alias);

    void SetAlias(std::string const &alias);

    void SetInputManager(InputManagerType inputStreamManager);

    void SetScheduler(int scheduler);

    void SetPreModule(bmf::BMFModule preModuleInstance);

    void AddCallback(long long key, bmf::BMFCallback callbackInstance);

    nlohmann::json Dump();

    std::shared_ptr<RealNode>
    AddModule(std::string const &alias, const bmf_sdk::JsonParam &option,
              std::vector<std::shared_ptr<RealStream>> inputStreams,
              std::string const &moduleName, ModuleType moduleType,
              std::string const &modulePath, std::string const &moduleEntry,
              InputManagerType inputStreamManager, int scheduler);

  private:
    friend bmf::builder::Graph;
    friend bmf::builder::Node;
    friend bmf::builder::Stream;
    friend RealGraph;
    friend RealStream;

    class ModuleMetaInfo {
      public:
        ModuleMetaInfo(std::string moduleName, ModuleType moduleType,
                       std::string modulePath, std::string moduleEntry);

        nlohmann::json Dump();

        std::string moduleName_;
        ModuleType moduleType_;
        std::string modulePath_;
        std::string moduleEntry_;
    };

    class NodeMetaInfo {
      public:
        NodeMetaInfo() = default;

        nlohmann::json Dump();

        unsigned int preModuleUID_ = 0;
        std::map<long long, unsigned int> callbackBinding_;
        std::shared_ptr<bmf::BMFModule> preModuleInstance_ = nullptr;
        std::map<long long, std::shared_ptr<bmf::BMFCallback>>
            callbackInstances_;
    };

    std::weak_ptr<RealGraph> graph_;
    int id_;
    std::string alias_;
    bmf_sdk::JsonParam option_;
    std::vector<std::shared_ptr<RealStream>> inputStreams_;
    std::vector<std::shared_ptr<RealStream>> outputStreams_;
    ModuleMetaInfo moduleInfo_;
    NodeMetaInfo metaInfo_;
    InputManagerType inputManager_;
    int scheduler_;

    std::map<std::string, std::shared_ptr<RealStream>> existedStreamNotify_;
};

class RealGraph : public std::enable_shared_from_this<RealGraph> {
  public:
    RealGraph(GraphMode runMode, const bmf_sdk::JsonParam &graphOption);

    RealGraph() = delete;

    RealGraph(RealGraph const &) = delete;

    RealGraph(RealGraph &&) = default;

    void GiveStreamAlias(std::shared_ptr<RealStream> stream,
                         std::string const &alias);

    void GiveNodeAlias(std::shared_ptr<RealNode> node,
                       std::string const &alias);

    nlohmann::json Dump();

    std::shared_ptr<RealNode>
    AddModule(std::string const &alias, const bmf_sdk::JsonParam &option,
              const std::vector<std::shared_ptr<RealStream>> &inputStreams,
              std::string const &moduleName, ModuleType moduleType,
              std::string const &modulePath, std::string const &moduleEntry,
              InputManagerType inputStreamManager, int scheduler);

    std::shared_ptr<RealNode> GetAliasedNode(std::string const &alias);

    std::shared_ptr<RealStream> GetAliasedStream(std::string const &alias);

    std::shared_ptr<RealStream> NewPlaceholderStream();

    void SetOption(const bmf_sdk::JsonParam &optionPatch);
    bmf::BMFGraph Instantiate(bool dumpGraph, bool needMerge);
    bmf::BMFGraph Instance();
    void Start(bool dumpGraph, bool needMerge);
    void Start(const std::vector<std::shared_ptr<internal::RealStream> >& streams,
                        bool dumpGraph, bool needMerge);

    int Run(bool dumpGraph, bool needMerge);
    Packet Generate(std::string streamName, bool block = true);
    int FillPacket(std::string stream_name, Packet packet, bool block = false);
    std::shared_ptr<RealStream> InputStream(std::string streamName, std::string notify, std::string alias);
  private:
    friend bmf::builder::Graph;
    friend bmf::builder::Node;
    friend bmf::builder::Stream;
    friend RealNode;
    friend RealStream;

    GraphMode mode_;
    std::vector<std::shared_ptr<RealStream>> inputStreams_;
    std::vector<std::shared_ptr<RealStream>> outputStreams_;
    std::vector<std::shared_ptr<RealNode>> nodes_;
    bmf_sdk::JsonParam graphOption_;

    std::shared_ptr<RealNode> placeholderNode_;
    std::shared_ptr<bmf::BMFGraph> graphInstance_ = nullptr;
    std::map<std::string, std::shared_ptr<RealStream>> existedStreamAlias_;
    std::map<std::string, std::shared_ptr<RealNode>> existedNodeAlias_;
};
} // namespace internal

std::string BMF_ENGINE_API GetVersion();

std::string BMF_ENGINE_API GetCommit();

void BMF_ENGINE_API ChangeDmpPath(std::string path);

bmf::BMFModule BMF_ENGINE_API GetModuleInstance(std::string const &moduleName,
                                                std::string const &option,
                                                ModuleType moduleType = Python,
                                                std::string const &modulePath = "",
                                                std::string const &moduleEntry = "");

bmf::BMFCallback BMF_ENGINE_API
GetCallbackInstance(std::function<bmf_sdk::CBytes(bmf_sdk::CBytes)> callback);

class BMF_ENGINE_API Stream {
  public:
    Stream() = delete;

    Stream(Stream const &) = default;

    Stream(Stream &&) = default;

  private:
    friend Node;
    friend Graph;

    explicit Stream(std::shared_ptr<internal::RealStream> baseP);
    std::shared_ptr<internal::RealStream> baseP_;

  public:
    void SetNotify(std::string const &notify);

    void SetAlias(std::string const &alias);
    void Start();

    Node
    Module(const std::vector<Stream> &inStreams, std::string const &moduleName,
           ModuleType moduleType, const bmf_sdk::JsonParam &option,
           std::string const &alias = "", std::string const &modulePath = "",
           std::string const &moduleEntry = "",
           InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node CppModule(
        const std::vector<Stream> &inStreams, std::string const &moduleName,
        const bmf_sdk::JsonParam &option, std::string const &alias = "",
        std::string const &modulePath = "", std::string const &moduleEntry = "",
        InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node PythonModule(
        const std::vector<Stream> &inStreams, std::string const &moduleName,
        const bmf_sdk::JsonParam &option, std::string const &alias = "",
        std::string const &modulePath = "", std::string const &moduleEntry = "",
        InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node GoModule(
        const std::vector<Stream> &inStreams, std::string const &moduleName,
        const bmf_sdk::JsonParam &option, std::string const &alias = "",
        std::string const &modulePath = "", std::string const &moduleEntry = "",
        InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node Decode(const bmf_sdk::JsonParam &decodePara,
                             std::string const &alias = "");

    Node EncodeAsVideo(const bmf_sdk::JsonParam &encodePara,
                                    std::string const &alias = "");

    Node EncodeAsVideo(Stream audioStream,
                                    const bmf_sdk::JsonParam &encodePara,
                                    std::string const &alias = "");

    Node FFMpegFilter(const std::vector<Stream> &inStreams,
                                   std::string const &filterName,
                                   bmf_sdk::JsonParam filterPara,
                                   std::string const &alias = "");

    template <typename T,
              typename std::enable_if<
                  std::is_integral<T>{} || std::is_floating_point<T>{} ||
                      std::is_convertible<T, std::string const &>{} ||
                      std::is_convertible<T, nlohmann::json>{},
                  bool>::type = true>
    Node FFMpegFilter(std::vector<Stream> inStreams,
                                   std::string const &filterName, T filterPara,
                                   std::string const &alias = "");

    template <typename T>
    Node Vflip(T para, std::string const &alias = "");

    template <typename T>
    Node Scale(T para, std::string const &alias = "");

    template <typename T>
    Node Setsar(T para, std::string const &alias = "");

    template <typename T>
    Node Pad(T para, std::string const &alias = "");

    template <typename T>
    Node Trim(T para, std::string const &alias = "");

    template <typename T>
    Node Setpts(T para, std::string const &alias = "");

    template <typename T>
    Node Loop(T para, std::string const &alias = "");

    template <typename T>
    Node Split(T para, std::string const &alias = "");

    template <typename T>
    Node Adelay(T para, std::string const &alias = "");

    template <typename T>
    Node Atrim(T para, std::string const &alias = "");

    template <typename T>
    Node Afade(T para, std::string const &alias = "");

    template <typename T>
    Node Asetpts(T para, std::string const &alias = "");

    template <typename T>
    Node Amix(std::vector<Stream> inStreams, T para,
                           std::string const &alias = "");

    template <typename T>
    Node Overlay(std::vector<Stream> inStreams, T para,
                              std::string const &alias = "");

    template <typename T>
    Node Concat(std::vector<Stream> inStreams, T para,
                             std::string const &alias = "");

    Node Fps(int fps, std::string const &alias = "");

    std::string GetName();

  private:
    Node ConnectNewModule(
        std::string const &alias, const bmf_sdk::JsonParam &option,
        const std::vector<Stream> &inputStreams, std::string const &moduleName,
        ModuleType moduleType, std::string const &modulePath,
        std::string const &moduleEntry, InputManagerType inputStreamManager,
        int scheduler);

    Node InternalFFMpegFilter(const std::vector<Stream> &inStreams,
                                           std::string const &filterName,
                                           const bmf_sdk::JsonParam &filterPara,
                                           std::string const &alias = "");
};

class BMF_ENGINE_API Node {
  public:
    Node() = delete;

    Node(Node const &) = default;

    Node(Node &&) = default;

  private:
    friend class Stream;

    friend Graph;

    explicit Node(std::shared_ptr<internal::RealNode> baseP);
    std::shared_ptr<internal::RealNode> baseP_;

  public:
    class Stream operator[](int index);

    class Stream operator[](std::string const &notifyOrAlias);

    class Stream Stream(int index);

    class Stream Stream(std::string const &notifyOrAlias);

    operator class Stream();

    void SetAlias(std::string const &alias);

    void
    SetInputStreamManager(InputManagerType inputStreamManager);

    void SetThread(int threadNum);

    void SetPreModule(const bmf::BMFModule &preModuleInstance);

    void AddCallback(long long key,
                                  const bmf::BMFCallback &callbackInstance);

    void Start();

    Node Module(
        const std::vector<class Stream> &inStreams,
        std::string const &moduleName, ModuleType moduleType,
        const bmf_sdk::JsonParam &option, std::string const &alias = "",
        std::string const &modulePath = "", std::string const &moduleEntry = "",
        InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node CppModule(const std::vector<class Stream> &inStreams,
                                std::string const &moduleName,
                                const bmf_sdk::JsonParam &option,
                                std::string const &alias = "",
                                std::string const &modulePath = "",
                                std::string const &moduleEntry = "",
                                InputManagerType inputStreamManager = Immediate,
                                int scheduler = 0);

    Node PythonModule(
        const std::vector<class Stream> &inStreams,
        std::string const &moduleName, const bmf_sdk::JsonParam &option,
        std::string const &alias = "", std::string const &modulePath = "",
        std::string const &moduleEntry = "",
        InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node GoModule(const std::vector<class Stream> &inStreams,
                               std::string const &moduleName,
                               const bmf_sdk::JsonParam &option,
                               std::string const &alias = "",
                               std::string const &modulePath = "",
                               std::string const &moduleEntry = "",
                               InputManagerType inputStreamManager = Immediate,
                               int scheduler = 0);

    Node Decode(const bmf_sdk::JsonParam &decodePara,
                             std::string const &alias = "");

    Node EncodeAsVideo(const bmf_sdk::JsonParam &encodePara,
                                    std::string const &alias = "");

    Node EncodeAsVideo(class Stream audioStream,
                                    const bmf_sdk::JsonParam &encodePara,
                                    std::string const &alias = "");

    Node FFMpegFilter(const std::vector<class Stream> &inStreams,
                                   std::string const &filterName,
                                   bmf_sdk::JsonParam filterPara,
                                   std::string const &alias = "");

    template <typename T,
              typename std::enable_if<
                  std::is_integral<T>{} || std::is_floating_point<T>{} ||
                      std::is_convertible<T, std::string const &>{} ||
                      std::is_convertible<T, nlohmann::json>{},
                  bool>::type = true>
    Node FFMpegFilter(std::vector<class Stream> inStreams,
                                   std::string const &filterName, T filterPara,
                                   std::string const &alias = "");

    template <typename T>
    Node Vflip(T para, std::string const &alias = "");

    template <typename T>
    Node Scale(T para, std::string const &alias = "");

    template <typename T>
    Node Setsar(T para, std::string const &alias = "");

    template <typename T>
    Node Pad(T para, std::string const &alias = "");

    template <typename T>
    Node Trim(T para, std::string const &alias = "");

    template <typename T>
    Node Setpts(T para, std::string const &alias = "");

    template <typename T>
    Node Loop(T para, std::string const &alias = "");

    template <typename T>
    Node Split(T para, std::string const &alias = "");

    template <typename T>
    Node Adelay(T para, std::string const &alias = "");

    template <typename T>
    Node Atrim(T para, std::string const &alias = "");

    template <typename T>
    Node Afade(T para, std::string const &alias = "");

    template <typename T>
    Node Asetpts(T para, std::string const &alias = "");

    template <typename T>
    Node Amix(std::vector<class Stream> inStreams, T para,
                           std::string const &alias = "");

    template <typename T>
    Node Overlay(std::vector<class Stream> inStreams, T para,
                              std::string const &alias = "");

    template <typename T>
    Node Concat(std::vector<class Stream> inStreams, T para,
                             std::string const &alias = "");

    Node Fps(int fps, std::string const &alias = "");

  private:
    Node ConnectNewModule(
        std::string const &alias, const bmf_sdk::JsonParam &option,
        const std::vector<class Stream> &inputStreams,
        std::string const &moduleName, ModuleType moduleType,
        std::string const &modulePath, std::string const &moduleEntry,
        InputManagerType inputStreamManager, int scheduler);

    Node InternalFFMpegFilter(
        const std::vector<class Stream> &inStreams,
        std::string const &filterName, const bmf_sdk::JsonParam &filterPara,
        std::string const &alias = "");
};

class BMF_ENGINE_API SyncPackets {
  public:
    SyncPackets() = default;
    void Insert(int streamId, std::vector<Packet> frames);
    std::vector<Packet> operator[](int);
    std::map<int, std::vector<Packet>> packets;
};

class BMF_ENGINE_API SyncModule {
  public:
    SyncModule() = default;
    std::vector<int> inputStreams;
    std::vector<int> outputStreams;
    std::shared_ptr<bmf_sdk::Module> moduleInstance = nullptr;
    std::map<int, std::vector<Packet>>
    ProcessPkts(std::map<int, std::vector<Packet>> inputPackets);
    SyncPackets ProcessPkts(SyncPackets pkts = SyncPackets());
    int32_t Process(bmf_sdk::Task task);
    int32_t SendEOF();
    int32_t Init();
    int32_t Close();
};

class BMF_ENGINE_API Graph {
  public:
    explicit Graph(GraphMode runMode,
                                bmf_sdk::JsonParam graphOption);

    explicit Graph(GraphMode runMode,
                                nlohmann::json graphOption = {});

    Graph() = delete;

    Graph(Graph const &rhs) = default;

    Graph(Graph &&rhs) = default;

  private:
    friend class Stream;

    friend Node;

    std::shared_ptr<internal::RealGraph> graph_;

  public:
    void SetTotalThreadNum(int num);

    Stream NewPlaceholderStream();

    Node GetAliasedNode(std::string const &alias);

    Stream GetAliasedStream(std::string const &alias);

    bmf::BMFGraph Instantiate(bool dumpGraph = true,
                                           bool needMerge = true);

    bmf::BMFGraph Instance();

    int Run(bool dumpGraph = true, bool needMerge = true);

    void Start(bool dumpGraph = true, bool needMerge = true);

    void Start(std::vector<Stream>& generateStreams, bool dumpGraph = true, bool needMerge = true);

    std::string Dump();

    Node
    Module(const std::vector<Stream> &inStreams, std::string const &moduleName,
           ModuleType moduleType, const bmf_sdk::JsonParam &option,
           std::string const &alias = "", std::string const &modulePath = "",
           std::string const &moduleEntry = "",
           InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node CppModule(
        const std::vector<Stream> &inStreams, std::string const &moduleName,
        const bmf_sdk::JsonParam &option, std::string const &alias = "",
        std::string const &modulePath = "", std::string const &moduleEntry = "",
        InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node PythonModule(
        const std::vector<Stream> &inStreams, std::string const &moduleName,
        const bmf_sdk::JsonParam &option, std::string const &alias = "",
        std::string const &modulePath = "", std::string const &moduleEntry = "",
        InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node GoModule(
        const std::vector<Stream> &inStreams, std::string const &moduleName,
        const bmf_sdk::JsonParam &option, std::string const &alias = "",
        std::string const &modulePath = "", std::string const &moduleEntry = "",
        InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    Node Decode(const bmf_sdk::JsonParam &decodePara,
                             std::string const &alias = "");

    Node Decode(const bmf_sdk::JsonParam &decodePara,
                             Stream controlStream,
                             std::string const &alias = "");

    Node Encode(Stream videoStream, Stream audioStream,
                             const bmf_sdk::JsonParam &encodePara,
                             std::string const &alias = "");

    Node Encode(Stream videoStream,
                             const bmf_sdk::JsonParam &encodePara,
                             std::string const &alias = "");

    Node Encode(const bmf_sdk::JsonParam &encodePara,
                             std::string const &alias = "");

    Node FFMpegFilter(const std::vector<Stream> &inStreams,
                                   std::string const &filterName,
                                   const bmf_sdk::JsonParam &filterPara,
                                   std::string const &alias = "");

    template <typename T,
              typename std::enable_if<
                  std::is_integral<T>{} || std::is_floating_point<T>{} ||
                      std::is_convertible<T, std::string const &>{} ||
                      std::is_convertible<T, nlohmann::json>{},
                  bool>::type = true>
    Node FFMpegFilter(std::vector<Stream> inStreams,
                                   std::string const &filterName, T filterPara,
                                   std::string const &alias = "");

    template <typename T>
    Node Vflip(Stream inStream, T para,
                            std::string const &alias = "");

    template <typename T>
    Node Scale(Stream inStream, T para,
                            std::string const &alias = "");

    template <typename T>
    Node Setsar(Stream inStream, T para,
                             std::string const &alias = "");

    template <typename T>
    Node Pad(Stream inStream, T para,
                          std::string const &alias = "");

    template <typename T>
    Node Trim(Stream inStream, T para,
                           std::string const &alias = "");

    template <typename T>
    Node Setpts(Stream inStream, T para,
                             std::string const &alias = "");

    template <typename T>
    Node Loop(Stream inStream, T para,
                           std::string const &alias = "");

    template <typename T>
    Node Split(Stream inStream, T para,
                            std::string const &alias = "");

    template <typename T>
    Node Adelay(Stream inStream, T para,
                             std::string const &alias = "");

    template <typename T>
    Node Atrim(Stream inStream, T para,
                            std::string const &alias = "");

    template <typename T>
    Node Afade(Stream inStream, T para,
                            std::string const &alias = "");

    template <typename T>
    Node Asetpts(Stream inStream, T para,
                              std::string const &alias = "");

    template <typename T>
    Node Amix(std::vector<Stream> inStreams, T para,
                           std::string const &alias = "");

    template <typename T>
    Node Overlay(std::vector<Stream> inStreams, T para,
                              std::string const &alias = "");

    template <typename T>
    Node Concat(std::vector<Stream> inStreams, T para,
                             std::string const &alias = "");

    Node Fps(Stream inStream, int fps,
                          std::string const &alias = "");

    SyncModule
    Sync(const std::vector<int> inStreams, const std::vector<int> outStreams,
         bmf_sdk::JsonParam moduleOption, std::string const &moduleName,
         ModuleType moduleType = CPP, std::string const &modulePath = "",
         std::string const &moduleEntry = "", std::string const &alias = "",
         InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    SyncModule
    Sync(const std::vector<int> inStreams, const std::vector<int> outStreams,
         nlohmann::json moduleOption, std::string const &moduleName,
         ModuleType moduleType = CPP, std::string const &modulePath = "",
         std::string const &moduleEntry = "", std::string const &alias = "",
         InputManagerType inputStreamManager = Immediate, int scheduler = 0);

    std::map<int, std::vector<Packet>>
    Process(SyncModule module, std::map<int, std::vector<Packet>> inputPackets);

    SyncPackets Process(SyncModule module,
                                     SyncPackets pkts = SyncPackets());

    int32_t  Init(SyncModule module);

    int32_t  Close(SyncModule module);

    int32_t  SendEOF(SyncModule module);

    void SetOption(const bmf_sdk::JsonParam &optionPatch);

    Packet Generate(std::string streamName, bool block = true);

    Stream InputStream(std::string streamName, std::string notify, std::string alias);

    int FillPacket(std::string stream_name, Packet packet, bool block = false);

  private:
    Node
    NewNode(std::string const &alias, const bmf_sdk::JsonParam &option,
            const std::vector<Stream> &inputStreams,
            std::string const &moduleName, ModuleType moduleType,
            std::string const &modulePath, std::string const &moduleEntry,
            InputManagerType inputStreamManager, int scheduler);
    Node InternalFFMpegFilter(const std::vector<Stream> &inStreams,
                                           std::string const &filterName,
                                           const bmf_sdk::JsonParam &filterPara,
                                           std::string const &alias = "");
};
} // namespace bmf::builder

#include "builder.ipp"

#endif // BMF_ENGINE_BUILDER_HPP
