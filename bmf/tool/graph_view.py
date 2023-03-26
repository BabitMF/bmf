import graphviz
import json
import sys


stream_color = '#99ccff'
node_color = '#ffcc00'


def parse_stream_name(stream):
    # get stream label and name
    if ':' in stream:
        ss = stream.split(":", 1)
        stream_label = ss[0]
        stream_key = ss[1]
    else:
        stream_label = None
        stream_key = stream
    return stream_key, stream_label


def find_all_edges_for_output_stream(stream, idx, in_node_config, bmf_graph, view_graph):
    stream = stream['identifier']
    # check for every node's every output
    for out_node_config in bmf_graph['nodes']:
        for input_idx, input_stream in enumerate(out_node_config['input_streams']):
            input_stream = input_stream['identifier']

            in_stream_key, in_stream_label = parse_stream_name(stream)
            out_stream_key, out_stream_label = parse_stream_name(input_stream)

            # it's an edge
            if in_stream_key == out_stream_key:
                in_stream_name = str(idx)
                # get output label if have
                if in_stream_label is not None:
                    in_stream_name = in_stream_label

                out_stream_name = str(input_idx)
                # get input label if have
                if out_stream_label is not None:
                    out_stream_name = out_stream_label

                edge_name = '%s:%s' % (in_stream_name, out_stream_name)

                if in_node_config is None:
                    view_graph.edge(stream, out_node_config['view_name'], edge_name)
                else:
                    view_graph.edge(in_node_config['view_name'], out_node_config['view_name'], edge_name)


def find_all_edges_for_input_stream(stream, idx, in_node_config, bmf_graph, view_graph):
    stream = stream['identifier']
    # check for every node's every output
    for out_node_config in bmf_graph['nodes']:
        for output_idx, output_stream in enumerate(out_node_config['output_streams']):
            output_stream = output_stream['identifier']

            in_stream_key, in_stream_label = parse_stream_name(output_stream)
            out_stream_key, out_stream_label = parse_stream_name(stream)

            # it's an edge
            if in_stream_key == out_stream_key:
                in_stream_name = str(output_idx)
                # get input label if have
                if in_stream_label is not None:
                    in_stream_name = in_stream_label

                out_stream_name = str(idx)
                # get output label if have
                if out_stream_label is not None:
                    out_stream_name = out_stream_label

                edge_name = '%s:%s' % (in_stream_name, out_stream_name)

                if in_node_config is None:
                    view_graph.edge(out_node_config['view_name'], stream, edge_name)
                else:
                    view_graph.edge(out_node_config['view_name'], in_node_config['view_name'], edge_name)


def view(graph_file):
    with open(graph_file, 'r') as f:
        bmf_graph = json.load(f)

    if bmf_graph is not None:
        view_graph = graphviz.Digraph(format='png')
        view_graph.attr(rankdir='LR')

        # create graph input stream
        for input_stream in bmf_graph['input_streams']:
            input_stream = input_stream['identifier']
            view_graph.node(
                input_stream, input_stream, shape='circle', style='filled', fillcolor=stream_color
            )

        # create graph output stream
        for output_stream in bmf_graph['output_streams']:
            output_stream = output_stream['identifier']
            view_graph.node(
                output_stream, output_stream, shape='circle', style='filled', fillcolor=stream_color
            )

        # create node
        for node_config in bmf_graph['nodes']:
            if 'name' in node_config['option']:
                node_name = node_config['option']['name']
            else:
                node_name = node_config['module_info']['name']

            node_name = '%s(%d)' % (node_name, node_config['id'])

            # store the view name for edge connection
            node_config['view_name'] = node_name

            view_graph.node(
                node_name, node_name, shape='box', style='filled', fillcolor=node_color
            )

        # create edge for nodes
        for in_node_config in bmf_graph['nodes']:
            # for every node output stream
            for output_idx, output_stream in enumerate(in_node_config['output_streams']):
                find_all_edges_for_output_stream(output_stream, output_idx, in_node_config, bmf_graph, view_graph)

        # create edge for graph input stream
        for input_stream in bmf_graph['input_streams']:
            find_all_edges_for_output_stream(input_stream, 0, None, bmf_graph, view_graph)

        # create edge for graph output stream
        for output_stream in bmf_graph['output_streams']:
            find_all_edges_for_input_stream(output_stream, 0, None, bmf_graph, view_graph)

        ff = graph_file.split("/")
        output_file = ff[len(ff) - 1].split(".", 1)[0]
        view_graph.view(output_file, cleanup=True)


if __name__ == '__main__':
    view(sys.argv[1])
