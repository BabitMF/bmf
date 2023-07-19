from .bmf_stream import stream_operator
from .bmf_node import BmfNode

## @ingroup pyAPI
## @defgroup mdFunc module functions
###@{
# BMF module related functions
###@}

bmf_modules = {
    'ff_decoder': 'c_ffmpeg_decoder',
    'ff_filter': 'c_ffmpeg_filter',
    'ff_encoder': 'c_ffmpeg_encoder',
    'pass_through': 'pass_through'
}


## @ingroup pyAPI
## @ingroup mdFunc
###@{
#  To build a BMF node by Args.
#  @param streams:
#  @param module_info: the module info of the module
#  @param option: the option for this module, for example:
#                  {
#                     'alias': 'pass_through',
#                     'output_path': output_path
#                  }
#  @param input_manager: immediate by default. It's the input stream manager of this module
#  @param pre_module: none by default. It's a previous CREATED module OBJECT by bmf.create_module()
#  @param scheduler: 0 by default. It's a dedicate thread to schedule this module
#  @return The stream object of the module
@stream_operator()
def module(streams,
           module_info,
           option=None,
           module_path="",
           entry="",
           input_manager='immediate',
           pre_module=None,
           scheduler=0,
           stream_alias=None):
    ###@}
    if option is None:
        option = {}
    if isinstance(module_info, str):
        return BmfNode(
            {
                "name": module_info,
                "type": "",
                "path": module_path,
                "entry": entry
            }, option, streams, input_manager, pre_module,
            scheduler).stream(stream_alias=stream_alias)
    return BmfNode(module_info, option, streams, input_manager, pre_module,
                   scheduler).stream(stream_alias=stream_alias)


## @ingroup pyAPI
## @ingroup mdFunc
###@{
#  To pass through the input stream packets to output (if connected, by sequence, 1:1)
#  @param streams: the input stream list of the module
#  @return Stream(s) of the module
@stream_operator()
def pass_through(stream, type="", path="", entry="", stream_alias=None):
    ###@}
    module_info = {
        "name": bmf_modules['pass_through'],
        "type": type,
        "path": path,
        "entry": entry
    }
    return BmfNode(module_info, {}, stream,
                   'immediate').stream(stream_alias=stream_alias)


## @ingroup pyAPI
## @ingroup mdFunc
###@{
#  The null sink module which will drop all the input from upstream
#  @param streams: the input stream list of the module
#  @return Stream(s) of the module
@stream_operator()
def null_sink(stream, type="", path="", entry=""):
    ###@}
    module_info = {
        "name": 'null_sink',
        "type": type,
        "path": path,
        "entry": entry
    }
    return BmfNode(module_info, {}, stream, input_manager='immediate')


@stream_operator()
def upload(stream, type="", path="", entry=""):
    module_info = {
        "name": 'upload',
        "type": type,
        "path": path,
        "entry": entry
    }
    return BmfNode(module_info, {}, stream, input_manager='immediate')


## @ingroup pyAPI
## @ingroup mdFunc
###@{
#  To build a python implemented module stream loaded by module library path and entry
#  @param streams: the input stream list of the module
#  @param name: the module name
#  @param option: the parameters for the module
#  @param module_path: the path to load the module
#  @param entry: the call entry of the module
#  @param input_manager: select the input manager for this module, immediate by default
#  @param pre_module: the previous created module object of this module
#  @return Stream(s) of the module
@stream_operator()
def py_module(streams,
              name,
              option=None,
              module_path="",
              entry="",
              input_manager='immediate',
              pre_module=None,
              scheduler=0,
              stream_alias=None):
    ###@}
    if option is None:
        option = {}
    return module(streams, {
        "name": name,
        "type": "python",
        "path": module_path,
        "entry": entry
    },
                  option,
                  input_manager=input_manager,
                  pre_module=pre_module,
                  scheduler=scheduler,
                  stream_alias=stream_alias)


## @ingroup pyAPI
## @ingroup mdFunc
###@{
#  To build a c/c++ implemented module stream loaded by module library path and entry
#  @param streams: the input stream list of the module
#  @param name: the module name
#  @param option: the parameters for the module
#  @param module_path: the path to load the module
#  @param entry: the call entry of the module
#  @param input_manager: select the input manager for this module, immediate by default
#  @param pre_module: the previous created module object of this module
#  @return Stream(s) of the module
@stream_operator()
def c_module(streams,
             name,
             module_path="",
             entry="",
             option=None,
             input_manager='immediate',
             pre_module=None,
             scheduler=0,
             stream_alias=None):
    ###@}
    if option is None:
        option = {}
    return module(streams, {
        "name": name,
        "type": "c++",
        "path": module_path,
        "entry": entry
    },
                  option,
                  input_manager=input_manager,
                  pre_module=pre_module,
                  scheduler=scheduler,
                  stream_alias=stream_alias)


@stream_operator()
def go_module(streams,
              name,
              module_path=None,
              entry=None,
              option=None,
              input_manager='immediate',
              pre_module=None,
              scheduler=0,
              stream_alias=None):
    if option is None:
        option = {}
    return module(streams, {
        "name": name,
        "type": "go",
        "path": module_path,
        "entry": entry
    },
                  option,
                  input_manager=input_manager,
                  pre_module=pre_module,
                  scheduler=scheduler,
                  stream_alias=stream_alias)
