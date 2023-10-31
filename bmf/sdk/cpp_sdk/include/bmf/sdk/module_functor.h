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

#include <bmf/sdk/module_manager.h>

namespace bmf_sdk {
/**
 * @brief Task.timestamp() == DONE or eof packet is returned from any output
 * stream
 *
 */
class BMF_API ProcessDone : public std::runtime_error {
  public:
    ProcessDone(const std::string &what) : std::runtime_error(what) {}
};

class BMF_API ModuleFunctor {
    struct Private;

  public:
    ModuleFunctor() = default;
    ModuleFunctor(const std::shared_ptr<Module> &m, int ninputs, int noutputs);
    ~ModuleFunctor();

    Module &module() const;

    //
    std::vector<Packet> operator()(const std::vector<Packet> &inputs);

    /**
     * @brief invoke module once with inputs, cleanup all un-fetched packet is
     *  cleanup is true
     * @param inputs
     * @param cleanup
     */
    ModuleFunctor &execute(const std::vector<Packet> &inputs,
                           bool cleanup = true);

    /**
     * @brief
     *
     * @param idx port index
     * @return std::vector<Packet>
     */
    std::vector<Packet> fetch(int idx);

    int ninputs() const;
    int noutputs() const;

    bool defined() const { return self.get() != nullptr; }

  private:
    std::shared_ptr<Private> self;
};

// type safe implementaion for ModuleFunctor
template <typename Inputs, typename Outputs> class ModuleFunctorSafe;

template <typename... IArgs, typename... OArgs>
class ModuleFunctorSafe<std::tuple<IArgs...>, std::tuple<OArgs...>> {
    ModuleFunctor impl;

  public:
    using Inputs = std::tuple<IArgs...>;
    using Outputs = std::tuple<OArgs...>;

    ModuleFunctorSafe(const std::shared_ptr<Module> &m)
        : impl(m, sizeof...(IArgs), sizeof...(OArgs)) {}

    Module &module() const { return impl.module(); }

    // interface for regular ouputs
    Outputs operator()(IArgs... args) {
        std::vector<Packet> inputs{Packet(args)...};
        auto outputs = impl(inputs);
        return cast_to_tuple(outputs,
                             std::make_index_sequence<sizeof...(OArgs)>());
    }

    // interface for irregular ouputs
    ModuleFunctorSafe &execute(IArgs... args, bool cleanup = true) {
        std::vector<Packet> inputs{Packet(args)...};
        impl.execute(inputs, cleanup);
        return *this;
    }

    template <int Port> auto fetch() {
        using T = typename IndexToType<Port, OArgs...>::type;
        std::vector<T> outs;
        for (auto &pkt : impl.fetch(Port)) {
            outs.push_back(pkt.template get<T>());
        }
        return outs;
    }

  protected:
    template <size_t Index, typename T, typename... Args> struct IndexToType {
        using type = typename IndexToType<Index - 1, Args...>::type;
    };

    template <typename T, typename... Args> struct IndexToType<0, T, Args...> {
        using type = T;
    };

    template <size_t... Index>
    Outputs cast_to_tuple(const std::vector<Packet> &outs,
                          std::index_sequence<Index...>) {
        return std::make_tuple(outs[Index].get<OArgs>()...);
    }
};

/**
 * @brief wrap a module as a functor, ref test_module_functor.cpp for details
 *
 * @tparam Inputs input types as tuple
 * @tparam Outputs output types as tuple
 * @param module
 * @return auto
 */
template <typename Inputs, typename Outputs>
auto make_sync_func(const std::shared_ptr<Module> &module) {
    return ModuleFunctorSafe<Inputs, Outputs>(module);
}

template <typename Inputs, typename Outputs>
auto make_sync_func(const ModuleInfo &info, const JsonParam &option = {},
                    int32_t node_id = 0) {
    auto &M = ModuleManager::instance();
    auto factory = M.load_module(info);
    if (factory == nullptr) {
        throw std::runtime_error("Load module " + info.module_name + " failed");
    }
    return make_sync_func<Inputs, Outputs>(factory->make(node_id, option));
}

BMF_API ModuleFunctor make_sync_func(const ModuleInfo &info, int32_t ninputs,
                                     int32_t noutputs,
                                     const JsonParam &option = {},
                                     int32_t node_id = 0);

} // namespace bmf_sdk