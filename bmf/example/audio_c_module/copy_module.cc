#include "copy_module.h"

using namespace boost::python;

void CopyModule::process(Task &task) {
    PacketQueueMap &input_queue_map = task.get_inputs();
    PacketQueueMap::iterator it;

    // process all input queues
    for (it = input_queue_map.begin(); it != input_queue_map.end(); it++) {
        // input stream label
        std::string label = it->first;

        // input packet queue
        std::queue<Packet> in_queue = it->second;

        // process all packets in one input queue
        while (!in_queue.empty()) {
            // Get a input packet
            Packet pkt = in_queue.front();
            in_queue.pop();

            // if packet is eof, set module done
            if (pkt.get_timestamp() == BMF_EOF) {
                task.set_timestamp(DONE);
                task.add_packet_to_out_queue(label, generate_eof_packet());
                return;
            }

            // Get packet data
            // Here we should know the data type in packet
            const AudioFrame &frame = pkt.get_data<AudioFrame>();

            // Create output frame
            AudioFrame *out_frame = new AudioFrame(frame.get_format(),frame.get_layout_name(),frame.get_samples());

            // Copy pts and time base from input frame to output frame
            out_frame->set_pts(frame.get_pts());
            out_frame->set_time_base(frame.get_time_base());
            out_frame->set_sample_rate(frame.get_sample_rate());
            // Copy data from input frame to output frame
            for (int i = 0; i < frame.get_plane_num(); i ++) {
                // Got one input and output data plane
                AudioPlane in_plane = frame.get_planes()[i];
                AudioPlane out_plane = out_frame->get_planes()[i];
                memcpy(out_plane.get_buffer(),
                       in_plane.get_buffer(),
                       frame.get_valid_byte_size());

            }

            // Add output frame to output queue
            Packet output_pkt;
            output_pkt.set_data<AudioFrame>(*out_frame);
            task.add_packet_to_out_queue(label, output_pkt);
        }
    }
}

BMF_MODULE(copy_module, CopyModule)
