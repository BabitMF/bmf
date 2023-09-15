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
#undef BMF_BUILD_SHARED

#include "module_factory.h"
#include <bmf/sdk/compat/path.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_manager.h>
#include <bmf/sdk/shared_library.h>
#include <nlohmann/json.hpp>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <string>

const fs::path module_root_path = bmf_sdk::s_bmf_repo_root;

int list_modules() {
    auto &M = bmf_sdk::ModuleManager::instance();
    auto known_modules = M.resolve_all_modules();

    for (const auto &[module_name, module_info] : known_modules) {
        std::cout << module_name << std::endl;
    }
    return 0;
}

int dump_module(std::string module_name) {
    auto &M = bmf_sdk::ModuleManager::instance();
    bmf_sdk::ModuleInfo info(module_name);
    std::shared_ptr<bmf_sdk::ModuleFactoryI> factory;
    bool get_module_register_info = false;
    try {
        factory = M.load_module(info, &info);
        get_module_register_info =
            factory->module_info(info); // XXX: will throw an exception when
                                        // dump a python non-module package
    } catch (const std::exception &e) {
        BMFLOG(BMF_ERROR) << "could not find module:" << module_name << ", "
                          << e.what() << std::endl;
        return -1;
    }

    int w = std::string("module_entry: ").size();
    if (get_module_register_info) {
        w = std::string("module_description: ").size();
    }
    std::cout << std::setw(w) << std::setfill(' ')
              << "module_name: " << info.module_name << std::endl;
    std::cout << std::setw(w) << std::setfill(' ')
              << "module_entry: " << info.module_entry << std::endl;
    std::cout << std::setw(w) << std::setfill(' ')
              << "module_type: " << info.module_type << std::endl;
    std::cout << std::setw(w) << std::setfill(' ')
              << "module_path: " << info.module_path << std::endl;
    if (get_module_register_info) {
        std::cout << std::setw(w) << std::setfill(' ')
                  << "module_description: " << info.module_description
                  << std::endl;
        std::cout << std::setw(w) << std::setfill(' ')
                  << "module_tag: " << info.module_tag << std::endl;
    }
    return 0;
}

int install_module(const bmf_sdk::ModuleInfo &info, bool force) {
    // check module_path
    auto original_path = fs::path(info.module_path);
    if (!fs::exists(original_path) || !fs::is_directory(original_path)) {
        BMFLOG(BMF_ERROR) << "invalid module_path:" << info.module_path
                          << std::endl;
        return -1;
    }

    // check module_type and module_entry
    // for c++ and python, check that the entry is correct.
    // for go, parse the entry to get the dynamic library name
    std::string module_file, _;
    auto &M = bmf_sdk::ModuleManager::instance();
    if (info.module_type == "c++" || info.module_type == "python" ||
        info.module_type == "go") {
        auto &M = bmf_sdk::ModuleManager::instance();
        try {
            std::tie(module_file, _) = M.parse_entry(info.module_entry, true);
        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << e.what() << std::endl;
            BMFLOG(BMF_ERROR)
                << "invalid module_entry:" << info.module_entry << std::endl;
            return -1;
        }

        // check module file
        if (info.module_type == "c++" || info.module_type == "go") {
            module_file =
                (fs::path(info.module_path) /
                 (module_file + bmf_sdk::SharedLibrary::default_extension()))
                    .string();
        } else {
            module_file =
                (fs::path(info.module_path) / (module_file + ".py")).string();
        }
        if (!fs::exists(module_file)) {
            BMFLOG(BMF_ERROR)
                << "cannot find the module file:" << module_file << std::endl;
            return -1;
        }
    } else {
        BMFLOG(BMF_ERROR) << "invalid module_type, must be one of c++/python/go"
                          << std::endl;
        return -1;
    }

    if (!fs::exists(module_root_path) &&
        !fs::create_directories(module_root_path)) {
        BMFLOG(BMF_ERROR) << "create module root directory failed."
                          << std::endl;
        return -1;
    }

    auto installed_path = module_root_path / ("Module_" + info.module_name);
    if (fs::exists(installed_path)) {
        if (force) {
            fs::remove_all(installed_path);
        } else {
            BMFLOG(BMF_WARNING)
                << info.module_name << " has already installed." << std::endl;
            return -1;
        }
    }
    fs::copy(original_path, installed_path);

    // bmf_sdk::JsonParam meta;
    nlohmann::json meta;
    meta["name"] = info.module_name;
    meta["revision"] = info.module_revision;
    meta["type"] = info.module_type;
    meta["entry"] = info.module_entry;
    // XXX: When "path" is empty, module_manager loader will automatically
    // calculate
    bmf_sdk::JsonParam(meta).store((installed_path / "meta.info").string());

    std::cout << "Installing the module:" << info.module_name << " in "
              << installed_path << " success." << std::endl;

    return 0;
}

int uninstall_module(std::string module_name) {
    auto module_path = module_root_path / ("Module_" + module_name);
    if (!fs::exists(module_path) || !fs::is_directory(module_path)) {
        BMFLOG(BMF_WARNING)
            << "Module:" << module_name << " is not exist" << std::endl;
        return -1;
    }
    fs::remove_all(module_path);
    std::cout << "Uninstalling the module:" << module_name << " from "
              << module_path << " success." << std::endl;
    return 0;
}

std::map<std::string, std::function<void()>> help_infos{
    {"help",
     []() { std::cout << "\thelp\t\tshow this help message" << std::endl; }},
    {"list",
     []() {
         std::cout << "\tlist\t\tlist all locally installed modules"
                   << std::endl;
     }},
    {"dump",
     []() {
         std::cout << "\tdump\t\t<module_name>" << std::endl;
         std::cout
             << "\t\t\tdump installed module infos specified by module_name"
             << std::endl;
     }},
    {"install",
     []() {
         std::cout
             << "\tinstall\t\t[-f] <module_name> <module_type> <module_entry> "
                "<module_path> [module_revision]"
             << std::endl;
         std::cout << "\t\t\tinstall the module to BMF installation path("
                   << module_root_path << ")." << std::endl;
         std::cout << "\t\t\tthe module_revision is optional, default is v0.0.1"
                   << std::endl;
     }},
    {"uninstall", []() {
         std::cout << "\tuninstall\t<module_name>" << std::endl;
         std::cout << "\t\t\tuninstall the module from BMF installation path"
                   << std::endl;
     }}};

void help(const char *name, const char *cmd) {
    std::cout << "Usage:" << std::endl;
    if (!cmd || help_infos.find(cmd) == help_infos.end()) {
        for (const auto &p : help_infos) {
            p.second();
        }
        return;
    }
    help_infos[cmd]();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        help(argv[0], nullptr);
        exit(EXIT_FAILURE);
    }

    if (char *log_level = getenv("BMF_LOG_LEVEL"); log_level) {
        char buf[256];
        sprintf(buf, "BMF_LOG_LEVEL=%s", log_level);
        putenv(buf);
    } else {
        putenv("BMF_LOG_LEVEL=WARNING");
    }
    configure_bmf_log();

    const std::string operation = argv[1];
    if (operation == "help") {
        const char *cmd = argc >= 3 ? argv[2] : nullptr;
        help(argv[0], cmd);
        return 0;
    } else if (operation == "list") {
        int ret = list_modules();
        if (ret < 0) {
            BMFLOG(BMF_ERROR) << "list modules failed." << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (operation == "dump") {
        if (argc < 3) {
            help(argv[0], operation.c_str());
            exit(EXIT_FAILURE);
        }
        const char *module_name = argv[2];
        int ret = dump_module(module_name);
        if (ret < 0) {
            BMFLOG(BMF_ERROR) << "dump module info failed." << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (operation == "install") {
        bool force = false;
        int arg_start = 2;
        if (argc < arg_start + 1) {
            help(argv[0], operation.c_str());
            exit(EXIT_FAILURE);
        }

        if (std::string(argv[arg_start]) == "-f") {
            force = true;
            arg_start++;
        }

        if (argc < arg_start + 4) {
            help(argv[0], operation.c_str());
            exit(EXIT_FAILURE);
        }

        const char *module_revision = "v0.0.1";
        if (argc >= arg_start + 5) {
            module_revision = argv[arg_start + 4];
        }
        bmf_sdk::ModuleInfo info(argv[arg_start], argv[arg_start + 1],
                                 argv[arg_start + 2], argv[arg_start + 3],
                                 module_revision);
        int ret = install_module(info, force);
        if (ret < 0) {
            BMFLOG(BMF_ERROR) << "install module failed." << std::endl;
            exit(EXIT_FAILURE);
        }
    } else if (operation == "uninstall") {
        if (argc < 3) {
            help(argv[0], operation.c_str());
            exit(EXIT_FAILURE);
        }
        const char *module_name = argv[2];
        int ret = uninstall_module(module_name);
        if (ret < 0) {
            BMFLOG(BMF_ERROR) << "uninstall module failed." << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        std::cerr << "Invalid operation:" << operation << "!" << std::endl;
        help(argv[0], nullptr);
        exit(EXIT_FAILURE);
    }
    return 0;
}
