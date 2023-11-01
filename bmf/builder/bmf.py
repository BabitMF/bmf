import json

from bmf.lib._bmf import engine
from .bmf_graph import BmfGraph


## @ingroup pyAPI
###@{
#  To provide a BMF graph
#  @param option: the option for the graph
#  @return An BMF graph
def graph(option=None):
    ###@}
    if option is None:
        option = {}
    return BmfGraph(option)


## @ingroup pyAPI
## @ingroup mdFunc
###@{
#  To create an object of the module, can be used to create the real module before the graph run
#  @param module_info: the module name
#  @param option: the option for the module
#  @return An module object
def create_module(module_info, option):
    ###@}
    if isinstance(module_info, str):
        return engine.Module(module_info, json.dumps(option), "", "", "")
    return engine.Module(module_info["name"], json.dumps(option),
                         module_info.get("type", ""),
                         module_info.get("path", ""),
                         module_info.get("entry", ""))


def get_module_file_dependencies(module_name):
    meta_file_name = "/opt/tiger/bmf_mods/Module_" + module_name + "/meta.info"
    with open(meta_file_name, mode='r') as f:
        config_dict = json.loads(f.read())
        if 'file_dependencies' in config_dict:
            file_dependencies = config_dict["file_dependencies"]
            return file_dependencies
