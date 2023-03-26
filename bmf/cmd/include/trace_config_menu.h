#ifndef BMF_SUITE_TRACE_CONFIG_MENU_H
#define BMF_SUITE_TRACE_CONFIG_MENU_H

#include "menu.h"

class TraceMenu : public Menu {
public:
    TraceMenu(std::string text) : Menu(text) {}
    TraceMenu& SetTraceSelection() {
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
    TraceMenu& SetPrintDisable() {
        callback = [this]() {
            config["trace_print"] = "export BMF_TRACE_PRINTING=DISABLE";
        };
        return *this;
    }
    TraceMenu& SetTracelogDisable() {
        callback = [this]() {
            config["trace_logging"] = "export BMF_TRACE_LOGGING=DISABLE";
        };
        return *this;
    }
    TraceMenu& SetTraceConfigSave() {
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

#endif //BMF_SUITE_TRACE_CONFIG_MENU_H