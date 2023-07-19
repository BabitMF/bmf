import copy
import json
from .bmf_node import BmfNode
from .bmf_stream import BmfStream


class BmfOptimizer:

    def __init__(self):
        pass

    @staticmethod
    def find_merged_link(links, stream):
        result = False
        link_remove = None
        pin = None
        for link in links:
            if link['stream'].get_identifier() == stream.get_identifier():
                pin = link['pin']
                result = True
                link_remove = link
                break

        if link_remove is not None:
            links.remove(link_remove)
        return result, pin

    @staticmethod
    def merge_two_node(n1, n2):
        for stream in n2.get_input_streams():
            n1.add_input_stream(stream)

        for stream in n2.get_output_streams():
            n1.add_output_stream(stream)

        for f in n2.get_option()['filters']:
            n1.get_option()['filters'].append(copy.deepcopy(f))

        removed_stream = []
        output_stream_keys = []
        for output_stream in n1.get_output_streams():
            output_stream_keys.append(output_stream.get_identifier())
        for stream in n1.get_input_streams():
            if stream.get_identifier() in output_stream_keys:
                removed_stream.append(stream)

                for i, f in enumerate(n1.get_option()['filters']):
                    if 'inputs' in f:
                        r, out_pin = BmfOptimizer.find_merged_link(
                            f['inputs'], stream)
                        if r:
                            filter_id = i
                            break

                for i, f in enumerate(n1.get_option()['filters']):
                    if 'outputs' in f:
                        r, in_pin = BmfOptimizer.find_merged_link(
                            f['outputs'], stream)
                        if r:
                            link = dict()
                            link['input_pin'] = in_pin
                            link['output_pin'] = out_pin
                            link['output_filter'] = filter_id
                            if 'links' not in f:
                                f['links'] = []
                            f['links'].append(link)

        for stream in removed_stream:
            for input_stream in n1.get_input_streams():
                if input_stream.get_identifier() == stream.get_identifier():
                    n1.get_input_streams().remove(input_stream)
            for output_stream in n1.get_output_streams():
                if output_stream.get_identifier() == stream.get_identifier():
                    n1.get_output_streams().remove(output_stream)

    @staticmethod
    def convert_filter_para(node_config):
        new_option = dict()

        new_option['filters'] = list()
        f = dict()
        new_option['filters'].append(f)

        f['inputs'] = list()
        f['outputs'] = list()
        for i, input_stream in enumerate(node_config.get_input_streams()):
            input_pin = dict()
            input_pin['stream'] = input_stream
            input_pin['pin'] = i
            f['inputs'].append(input_pin)

        for i, output_stream in enumerate(node_config.get_output_streams()):
            output_pin = dict()
            output_pin['stream'] = output_stream
            output_pin['pin'] = i
            f['outputs'].append(output_pin)

        f['name'] = node_config.get_option()['name']

        if 'para' in node_config.get_option():
            f['para'] = node_config.get_option()['para']

        node_config.set_option(new_option)

    @staticmethod
    def convert_filter_para_for_graph(nodes):
        for node in nodes:
            if node.get_module_info().get_name() == 'c_ffmpeg_filter':
                BmfOptimizer.convert_filter_para(node)

    @staticmethod
    def replace_stream_name_with_id(node_config):
        if node_config is not None:
            for i, input_stream in enumerate(node_config.get_input_streams()):
                for f in node_config.get_option()['filters']:
                    if 'inputs' in f:
                        for input_pin in f['inputs']:
                            if not isinstance(
                                    input_pin['stream'], int
                            ) and input_pin['stream'].get_identifier(
                            ) == input_stream.get_identifier():
                                input_pin['stream'] = i
                                break
            for i, output_stream in enumerate(
                    node_config.get_output_streams()):
                for f in node_config.get_option()['filters']:
                    if 'outputs' in f:
                        for output_pin in f['outputs']:
                            if not isinstance(
                                    output_pin['stream'], int
                            ) and output_pin['stream'].get_identifier(
                            ) == output_stream.get_identifier():
                                output_pin['stream'] = i
                                break

    @staticmethod
    def replace_stream_name_for_graph(nodes):
        for node in nodes:
            if node.get_module_info().get_name() == 'c_ffmpeg_filter':
                BmfOptimizer.replace_stream_name_with_id(node)

    @staticmethod
    def merge_ffmpeg_filter_nodes(merge_nodes):
        if merge_nodes is None or len(merge_nodes) == 0:
            return

        merged_node = copy.deepcopy(merge_nodes[0])

        # for every nodes to merge
        i = 1
        while i < len(merge_nodes):
            BmfOptimizer.merge_two_node(merged_node, merge_nodes[i])
            i += 1

        BmfOptimizer.replace_stream_name_with_id(merged_node)

        return merged_node

    class Neighbour:

        def __init__(self):
            self.root_stream = None
            self.node = None

    @staticmethod
    def find_all_neighbours(opt_nodes, merged_node, root_stream):
        neighbours = []
        for output_stream in merged_node.get_output_streams():
            for node in opt_nodes:
                if output_stream in node.get_input_streams():
                    nb = BmfOptimizer.Neighbour()
                    nb.node = node
                    nb.root_stream = output_stream
                    neighbours.append(nb)
        return neighbours

    @staticmethod
    def has_circle(opt_nodes, merged_node, rec_stack, root_stream):
        rec_stack[merged_node.get_id()] = True

        neighbours = BmfOptimizer.find_all_neighbours(opt_nodes, merged_node,
                                                      root_stream)
        for nb in neighbours:
            if not rec_stack.get(nb.node.get_id(), False):
                ret, circle_stream = BmfOptimizer.has_circle(
                    opt_nodes, nb.node, rec_stack, root_stream)
                if ret:
                    return True, circle_stream
            else:
                return True, nb.root_stream

        rec_stack[merged_node.get_id()] = False
        return False, None

    @staticmethod
    def find_first_circle_node(opt_nodes, merged_node):
        rec_stack = dict()

        ret, stream = BmfOptimizer.has_circle(opt_nodes, merged_node,
                                              rec_stack, None)
        if ret:
            # print('has circle, stream:%s' % stream)
            return stream
        else:
            # print('no circle')
            return None

    @staticmethod
    def optimize(nodes, optimize=True):
        # convert filter para to new format
        BmfOptimizer.convert_filter_para_for_graph(nodes)

        if optimize:
            # nodes_done is used to record ffmpeg_filter node that already optimized
            nodes_done = []
            while True:
                nodes_to_merge = []
                # put all ffmpeg_filter nodes into nodes_to_merge and try to combine it to one node
                for node in nodes:
                    if node.get_module_info().get_name(
                    ) == 'c_ffmpeg_filter' and node not in nodes_done:
                        nodes_to_merge.append(node)
                for node in nodes_to_merge:
                    nodes.remove(node)

                # no node to optimize, quit
                if len(nodes_to_merge) == 0:
                    break

                while True:
                    # do node merge
                    merged_node = BmfOptimizer.merge_ffmpeg_filter_nodes(
                        nodes_to_merge)
                    if merged_node is not None:
                        nodes.append(merged_node)

                    # check if has a circle
                    circle_stream = BmfOptimizer.find_first_circle_node(
                        nodes, merged_node)
                    if circle_stream is not None:
                        # find circle-end node according to stream
                        for node in nodes_to_merge:
                            if circle_stream in node.get_input_streams():
                                circle_node = node
                                break

                        # remove it from nodes_to_merge and add it back to node list
                        nodes_to_merge.remove(circle_node)
                        nodes.append(circle_node)

                        nodes.remove(merged_node)
                        continue
                    else:
                        # this merged node is done, avoid repeatedly processing it
                        nodes_done.append(merged_node)
                        break

        # replace stream name with stream id in filter option
        BmfOptimizer.replace_stream_name_for_graph(nodes)
