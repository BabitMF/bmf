#!/usr/bin/env python

import os
import subprocess
import sys
import site
import glob
import re
from pathlib import PurePath


def find_in_python_root(filepath: str):
    filefullpath = ""
    found = False
    for package_root in site.getsitepackages():
        # Find the path of the real executable.
        filefullpath = os.path.join(package_root, filepath)

        if os.path.exists(filefullpath):
            found = True
            return found, filefullpath

    return found, ""


def exec_cmd(cmd: str, args):
    found, cmdpath = find_in_python_root(cmd)
    if found:
        # Execute it with all command line arguments.
        subprocess.call([cmdpath] + args)
    else:
        print("cannot find {}".format(os.path.basename(__file__)))
        exit(1)


def run_bmf_graph():
    exec_cmd(os.path.join("bmf", "bin", "run_bmf_graph"), sys.argv[1:])


def trace_format_log():
    exec_cmd(os.path.join("bmf", "bin", "trace_format_log"), sys.argv[1:])


def module_manager():
    exec_cmd(os.path.join("bmf", "bin", "module_manager"), sys.argv[1:])


def bmf_env():
    found, package_path = find_in_python_root("bmf")
    if found:
        print(
            "please set environments to find BMF libraries, maybe you need to add it to you .bashrc or .zshrc:\n"
        )
        print("export C_INCLUDE_PATH=${{C_INCLUDE_PATH}}:{}".format(
            os.path.join(package_path, "include")))
        print("export CPLUS_INCLUDE_PATH=${{CPLUS_INCLUDE_PATH}}:{}".format(
            os.path.join(package_path, "include")))
        print("export LIBRARY_PATH=${{LIBRARY_PATH}}:{}".format(
            os.path.join(package_path, "lib")))
        print("export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH}}:{}".format(
            os.path.join(package_path, "lib")))
