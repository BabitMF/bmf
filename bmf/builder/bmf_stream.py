import os
from bmf import Log, LogLevel

## @ingroup pyAPI
## @defgroup strClass BmfStream
###@{
# BMF stream class
###@}


class BmfStream:

    def __init__(self, stream_name, parent, notify, stream_alias=''):
        from .bmf_node import BmfNode
        from .bmf_graph import BmfGraph

        # while the stream is graph input stream, parent is graph
        # otherwise, parent is node it's attached
        self.graph_ = None
        if isinstance(parent, BmfNode):
            self.node_ = parent
        elif isinstance(parent, BmfGraph):
            self.graph_ = parent
            self.node_ = None

        self.notify_ = notify
        self.name_ = stream_name
        self.stream_alias_ = stream_alias

    def get_name(self):
        return self.name_

    def get_alias(self):
        return self.stream_alias_

    def get_notify(self):
        return self.notify_

    def get_node(self):
        return self.node_

    def get_graph(self):
        if self.graph_ is not None:
            return self.graph_
        elif self.node_ is not None:
            return self.node_.get_graph()
        return None

    def set_scheduler(self, scheduler):
        self.node_.set_scheduler(scheduler)

    def get_identifier(self):
        if isinstance(self.notify_, int):
            return self.name_
        else:
            return self.notify_ + ":" + self.name_

    def __getitem__(self, item):
        if self.node_ is not None:
            return self.get_node()[item]
        return None

    def stream(self, item):
        if self.node_ is not None:
            return self.get_node()[item]
        return None

    ## @ingroup pyAPI
    ## @ingroup strClass
    ###@{
    #  Using the stream of the module to call the routine of graph generate_config
    def generate_config_file(self, file_name="original_graph.json"):
        ###@}
        return self.node_.get_graph().generate_config_file(self,
                                                           file_name=file_name)

    ## @ingroup pyAPI
    ###@{
    #  Using the stream of the module to call the routine of graph run
    def run(self):
        ###@}
        return self.node_.get_graph().run(self)

    ## @ingroup pyAPI
    ## @ingroup strClass
    ###@{
    #  Using the stream object to call graph run without block
    def run_wo_block(self):
        return self.node_.get_graph().run_wo_block(self)

    ###@}

    def generateConfig(self, file_name):
        return self.node_.get_graph().generateConfig(file_name)

    def start(self):
        return self.node_.get_graph().start(self)

    def output_stream(self):
        self.get_graph().node_streams_.append(self)

    def server(self, mode=0):
        self.get_graph().node_streams_.append(self)
        if mode == 0:
            from bmf import ServerGateway
            server_gateway = ServerGateway(self.get_graph())
        elif mode == 1:
            from bmf import ServerGatewayNew
            server_gateway = ServerGatewayNew(self.get_graph())
        else:
            Log.log(LogLevel.DEBUG, "incorrect server mode")
            os._exit(1)
        server_gateway.init()
        return server_gateway


def stream_operator(name=None):

    def decorator(func):
        func_name = name or func.__name__
        setattr(BmfStream, func_name, func)
        return func

    return decorator
