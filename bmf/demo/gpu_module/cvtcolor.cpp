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

#include <bmf/sdk/bmf.h>
#include <bmf/sdk/packet.h>
#include <hmp/imgproc.h>

USE_BMF_SDK_NS

class TestCvtColorModule : public Module {
  public:
    TestCvtColorModule(int node_id, JsonParam option)
        : Module(node_id, option) {}

    ~TestCvtColorModule() {}

    virtual int process(Task &task);
};

int TestCvtColorModule::process(Task &task) {
    PacketQueueMap &input_queue_map = task.get_inputs();
    PacketQueueMap::iterator it;

    // process all input queues
    for (it = input_queue_map.begin(); it != input_queue_map.end(); it++) {
        // input stream label
        int label = it->first;

        // input packet queue
        Packet pkt;
        // process all packets in one input queue
        while (task.pop_packet_from_input_queue(label, pkt)) {
            // Get a input packet

            // if packet is eof, set module done
            if (pkt.timestamp() == BMF_EOF) {
                task.set_timestamp(DONE);
                task.fill_output_packet(label, Packet::generate_eof_packet());
                return 0;
            }

            // Get packet data
            // Here we should know the data type in packet
            auto vframe = pkt.get<VideoFrame>();
            if (vframe.device() == kCPU) {
                vframe = vframe.cuda();
            }
            // yuv2yuv
            hmp::PixelInfo nv12info{hmp::PF_NV12, hmp::CS_BT470BG,
                                    hmp::CR_MPEG};
            VideoFrame vframe_nv12{vframe.width(), vframe.height(), nv12info,
                                   kCUDA};
            hmp::TensorList nv12_tensor = vframe_nv12.frame().data();
            hmp::img::yuv_to_yuv(nv12_tensor, vframe.frame().data(),
                                 vframe_nv12.frame().pix_info(),
                                 vframe.frame().pix_info());

            // yuv2rgb
            hmp::PixelInfo cinfo{hmp::PF_RGB24, hmp::CS_BT709, hmp::CR_MPEG};
            hmp::PixelInfo pinfo{hmp::PF_YUV420P, hmp::CS_BT709, hmp::CR_MPEG};
            VideoFrame vframe_out{vframe.width(), vframe.height(), cinfo,
                                  kCUDA};
            hmp::Tensor out_tensor = vframe_out.frame().plane(0);
            hmp::img::yuv_to_rgb(out_tensor, vframe.frame().data(),
                                 vframe.frame().pix_info(), hmp::kNHWC);

            // Add output frame to output queue
            vframe_nv12.copy_props(vframe);
            vframe_out.copy_props(vframe);
            auto output_pkt = Packet(vframe_nv12);

            task.fill_output_packet(label, output_pkt);
        }
    }
    return 0;
}
REGISTER_MODULE_CLASS(TestCvtColorModule)
