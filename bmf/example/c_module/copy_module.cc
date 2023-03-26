#include "copy_module.h"

int CopyModule::process(Task &task) {
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

            // Deep copy
            VideoFrame vframe_out;
            if(vframe.is_image()){
                vframe_out = VideoFrame(vframe.image().clone());
            }
            else{
                vframe_out = VideoFrame(vframe.frame().clone());
            }
            vframe_out.copy_props(vframe);

            // Add output frame to output queue
            auto output_pkt = Packet(vframe_out);

            task.fill_output_packet(label, output_pkt);
        }
    }
    return 0;
}
REGISTER_MODULE_CLASS(CopyModule)
