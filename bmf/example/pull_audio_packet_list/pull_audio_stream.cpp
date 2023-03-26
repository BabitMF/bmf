#include "pull_audio_stream.h"
#include "bmf_av_packet_list.h"
#include <fstream>

using namespace boost::python;

PullAudioStreamModule::PullAudioStreamModule(int node_id, JsonParam option) : Module(node_id, option) {
    std::vector<JsonParam> data_param_list;
    option.get_object_list("data", data_param_list);

    for (int i = 0; i < data_param_list.size(); i++) {
        DataSource *p_data_source = new DataSource();
        data_param_list[i].get_string("data_path", p_data_source->data_path);
        data_param_list[i].get_string("size_path", p_data_source->size_path);
        p_data_source->f_data.open(p_data_source->data_path, std::ios::in | std::ios::binary);
        p_data_source->f_size.open(p_data_source->size_path);
        source_list.push_back(p_data_source);
    }
    return;
}

int PullAudioStreamModule::process(Task &task) {
    BMFAVPacketList bmf_av_packet_list;
    bmf_av_packet_list.set_pts(pts_);
    int index = 0;
    Packet packet;
    for (int i = 0; i < source_list.size(); i++) {
        DataSource *data_source = source_list[i];
        if (not data_source->f_size.eof()) {
            int packet_size;
            data_source->f_size >> packet_size;
            std::cout<<"packet_size:"<<packet_size<<std::endl;
            BMFAVPacket pkt = BMFAVPacket(NULL, packet_size);
//            pkt.set_pts(pts_);
            if (packet_size > 0) {
                data_source->f_data.read((char *) (pkt.get_data()), packet_size);
            }
            bmf_av_packet_list.push_av_packet(index, pkt);
            index++;
        }
    }
    bmf_av_packet_list.set_pts(pts_);
    if (bmf_av_packet_list.get_size() == 0) {
        task.fill_output_packet(0, Packet::generate_eof_packet());
        std::cout<<"pull end"<<std::endl;
        task.set_timestamp(DONE);
        return 0;
    }
    std::cout<<"pull audio process task pts:"<<pts_<<std::endl;
    pts_ = pts_ + 20;
    packet.set_data(bmf_av_packet_list);
    packet.set_data_class_type(PACKET_TYPE);
    packet.set_class_name("libbmf_module_sdk.BMFAVPacketList");
    packet.set_data_type(DATA_TYPE_C);
    timestamp_++;
    packet.set_timestamp(timestamp_);
    task.fill_output_packet(0, packet);

    return 0;
}

REGISTER_MODULE_CLASS(PullAudioStreamModule)
