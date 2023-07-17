import abc

### @defgroup PyMdSDK Python Module SDK

## @ingroup PyMdSDK
## @defgroup pyMdClass Module
###@{
# Module class in Python Module SDK
###@}

## @ingroup PyMdSDK
## @defgroup taskClass Task
###@{
# Task class in Python Module SDK
###@}

## @ingroup PyMdSDK
## @defgroup videoFrame VideoFrame
###@{
# video frame data holder, the VideoFrame class in Python Module SDK
###@}

## @ingroup PyMdSDK
## @defgroup audioFrame AudioFrame
###@{
# audio frame data holder, the AudioFrame class in Python Module SDK
###@}

## @ingroup PyMdSDK
## @defgroup videoPlane VideoPlane
###@{
# video plane data holder, the VideoPlane class in Python Module SDK
###@}

## @ingroup PyMdSDK
## @defgroup pktMd Packet
###@{
# the Packet class in Python Module SDK
###@}

## @ingroup PyMdSDK
## @defgroup dataQ DataQueue
###@{
# the queue of packet in Python Module SDK
###@}


class ProcessResult:
    OK = 0
    STOP = 1
    ERROR = 2


class InputType:
    VIDEO = 1
    PICTURELIST = 2
    VIDEOLIST = 3


class Module:
    """
    Module is base unit of processing data, it could have multiple
    input streams and multiple output streams, every module should
    implement its process() function, process() accept one or more
    packets one time and generate zero, one or more packets
    """
    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief construct bmf module
    #  @param node_id unique id .
    #  @param json_param json param of module.
    #  @return An module object
    @abc.abstractmethod
    def __init__(self, node=None, option=None):
        ###@}
        self.node_ = node
        return

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief init module
    def init(self):
        ###@}
        return 0

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief reset module when the module need to be reseted
    def reset(self):
        ###@}
        pass

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief dynamic_reset module when the option need to be updated.
    #  @param opt_reset dict value of option
    def dynamic_reset(self, opt_reset=None):
        ###@}
        pass

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief set node id of this module
    #  @param node node id of the module
    def set_node(self, node):
        ###@}
        self.node_ = node

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief check the module is subgraph
    #  @return true if the module is subgraph, else is false.
    def is_subgraph(self):
        ###@}
        return False

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief process task
    #  @param task reference to the @ref taskClass class. The module should process input packet in task and produce output packet to the task
    #  @return 0 is success, else is failed
    @abc.abstractmethod
    def process(self, task):
        ###@}
        return

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief close module
    def close(self):
        ###@}
        pass

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief check the module's input stream should hungry check
    #  @param input_idx input stream id
    #  @return true if need check , else is false
    def need_hungry_check(self, input_idx):
        ###@}
        return False

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief check the module's input stream need data
    #  @param input_idx input stream id
    #  @return true if need data , else is false
    def is_hungry(self, input_idx):
        ###@}
        return True

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief check the module is infinity
    #  @return true if infinity module , else is false
    def is_infinity(self):
        ###@}
        return False

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief set the graph callback to the module
    #  @param callback graph callback.
    def set_callback(self, callback):
        ###@}
        self.callback_ = callback

    ## @ingroup PyMdSDK
    ## @ingroup pyMdClass
    ###@{
    #  @brief get the graph config of the module
    #  @return graph config
    def get_graph_config(self):
        ###@}
        return None
