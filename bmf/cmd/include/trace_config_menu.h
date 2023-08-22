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
#ifndef BMF_SUITE_TRACE_CONFIG_MENU_H
#define BMF_SUITE_TRACE_CONFIG_MENU_H

#include "menu.h"

class TraceMenu : public Menu {
  public:
    TraceMenu(std::string text) : Menu(text) {}
    TraceMenu &SetTraceSelection() {
        callback = [this]() {
            std::string state;
            for (int i : selected_options) {
                if (this->options[i].option_tag == TAG_SELECT_ALL) {
                    config["trace_types"] = "export BMF_TRACE=ENABLE";
                    return;
                }
                if (state.empty())
                    state += "export BMF_TRACE=";
                else
                    state += ",";

                state += this->options[i].title;
            }
            if (state.empty())
                config.erase("trace_types");
            else
                config["trace_types"] = state;
        };
        return *this;
    }
    TraceMenu &SetPrintDisable() {
        callback = [this]() {
            config["trace_print"] = "export BMF_TRACE_PRINTING=DISABLE";
        };
        return *this;
    }
    TraceMenu &SetTracelogDisable() {
        callback = [this]() {
            config["trace_logging"] = "export BMF_TRACE_LOGGING=DISABLE";
        };
        return *this;
    }
    TraceMenu &SetTraceConfigSave() {
        callback = [this]() {
            std::ofstream envfile("env.sh");
            envfile << "#!/bin/sh" << std::endl;
            if (config.count("trace_types"))
                envfile << config["trace_types"] << std::endl;
            if (config.count("trace_print"))
                envfile << config["trace_print"] << std::endl;
            if (config.count("trace_logging"))
                envfile << config["trace_logging"] << std::endl;
        };
        return *this;
    }
};

#endif // BMF_SUITE_TRACE_CONFIG_MENU_H