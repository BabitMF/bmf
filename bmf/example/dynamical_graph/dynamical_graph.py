import sys
import time
import unittest
import threading
import json

sys.path.append("../../../")
import bmf
def test_dynmaical_reset():
    input_video_path = '../files/img.mp4'
    output_path = "./output.mp4"

    main_graph = bmf.graph()
    video1 = main_graph.decode({'input_path': input_video_path, 'alias': "decoder0"})

    passthru = bmf.module([video1['video'], video1['audio']], 'reset_pass_through',
                          {
                             "alias": "reset_pass_through",
                          },
                          "", "", "immediate")

    #instead of run() block function, here use none-blocked run
    passthru.run_wo_block()
    time.sleep(0.02)

    update_graph = bmf.graph()
    update_graph.dynamic_reset(
                               {
                                   'alias': 'reset_pass_through',
                                   'output_path': output_path,
                                   'video_params': {
                                       'codec': 'h264',
                                       'width': 320,
                                       'height': 240,
                                       'crf': 23,
                                       'preset': 'veryfast'
                                   }
                               }
                              )
    main_graph.update(update_graph)

    main_graph.close()

def test_dynmaical_graph():
    input_video_path = '../files/img.mp4'
    input_video_path2 = '../files/img.mp4'
    output_path = "./output.mp4"

    main_graph = bmf.graph()
    video1 = main_graph.decode({'input_path': input_video_path, 'alias': "decoder0"})

    passthru = bmf.module([video1['video'], video1['audio']], 'pass_through',
               {
                   'alias': "pass_through",
                },
               "", "", "immediate")

    #instead of run() block function, here use none-blocked run
    passthru.run_wo_block()
    time.sleep(0.1)

    #dynamic add a decoder which need output connection
    update_decoder = bmf.graph()
    video2 = update_decoder.decode(
                          {
                              'input_path': input_video_path2,
                              'alias': "decoder1"
                          })

    outputs = {'alias': 'pass_through', 'streams': 2}
    update_decoder.dynamic_add(video2, None, outputs)
    main_graph.update(update_decoder)
    time.sleep(0.03)

    #dynamic add a encoder which need input connection
    update_encoder = bmf.graph()
    encode = bmf.encode(None, None,
             {
                'output_path': output_path,
                'alias': "encoder1"
             })
    inputs = {'alias': 'pass_through', 'streams': 2}
    encode.get_graph().dynamic_add(encode, inputs, None)
    main_graph.update(encode.get_graph())
    time.sleep(0.05)

    #dynamic remove a decoder/encoder/pass_through
    remove_graph = bmf.graph()
    remove_graph.dynamic_remove({'alias': 'decoder1'})
    #remove_graph.dynamic_remove({'alias': 'pass_through'})
    #remove_graph.dynamic_remove({'alias': 'encoder1'})
    main_graph.update(remove_graph)

    time.sleep(2)
    main_graph.force_close()
    #main_graph.close()

actions = list()
class dy_action:
    def __init__(self):
        self.action = ""
        self.alias = ""

def action_thread(main_graph):
    global actions
    input_video_path2 = '../files/img.mp4'
    output_path = "./cb.mp4"
    count = 0;
    print("====== action thread started ====== ")
    while count < 2:
        if len(actions) > 0:
            dy = actions[0]
            actions.pop(0)
            print("====== action thread: ", dy.action, ": ", dy.alias)
            count += 1
            alias = dy.alias
            if dy.action == 'add_d':
                update_decoder = bmf.graph()
                video2 = update_decoder.decode(
                                      {
                                          'input_path': input_video_path2,
                                          'alias': alias
                                      })

                outputs = {'alias': 'callback_module', 'streams': 2}
                update_decoder.dynamic_add(video2, None, outputs)
                main_graph.update(update_decoder)

            # dynamical add encoder will generate some encode errors, won't use encoder temporary
            #if dy.action == 'add_e':
            #    update_encoder = bmf.graph()
            #    encode = bmf.encode(None, None,
            #             {
            #                'output_path': output_path,
            #                'alias': alias
            #             })
            #    inputs = {'alias': 'callback_module', 'streams': 2}
            #    encode.get_graph().dynamic_add(encode, inputs, None)
            #    main_graph.update(encode.get_graph())

            if dy.action == 'remove':
                remove_graph = bmf.graph()
                remove_graph.dynamic_remove({'alias': alias})
                main_graph.update(remove_graph)
        else:
            time.sleep(0.1)

def test_dynamical_graph_cb():
    input_video_path = "../files/img.mp4"

    # create graph
    main_graph = bmf.graph()
    def cb(para): # NOTICE: the callback should NOT do the real dynamical action to avoid module block/hang issue
        print("jpara decode", para.decode("utf-8"))
        jpara = json.loads(para.decode("utf-8"))
        global actions
        print("======call back======")
        for action in jpara.items():
            print(action[0], ": ", action[1])
            dy = dy_action()
            dy.action = action[0]
            dy.alias = action[1]
            actions.append(dy)
        print("======call back end======")

        return bytes("OK", "utf-8")

    thread = threading.Thread(target=action_thread, args=(main_graph,))
    thread.start()

    input_video_path = '../files/img.mp4'
    output_path = "./output.mp4"

    video1 = main_graph.decode({'input_path': input_video_path, 'alias': "decoder0"})

    passthru = bmf.module([video1['video'], video1['audio']], 'callback_module',
               {
                   'alias': "callback_module",
                },
               "", "", "immediate")
    passthru.get_node().add_user_callback(0, cb)

    passthru.run_wo_block()

    thread.join()
    time.sleep(3)
    main_graph.force_close()

if __name__ == '__main__':
    test_dynmaical_graph()
    test_dynamical_graph_cb()
    test_dynmaical_reset()
