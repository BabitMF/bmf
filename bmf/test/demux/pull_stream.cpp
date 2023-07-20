#include "pull_stream.h"
#include "bmf_av_packet.h"
#include <fstream>

using namespace boost::python;

PullStreamModule::PullStreamModule(int node_id, JsonParam option) : Module(node_id, option) {
    std::string data_file_path;
    option.get_string("data_path",data_file_path);
    std::string size_file_path;
    option.get_string("size_path",size_file_path);
    if (option.has_key("pts_path"))
    {
        option.get_string("pts_path",pts_file_path_);
        f_pts.open(pts_file_path_);
    }
    f_data.open(data_file_path,
                                std::ios::in | std::ios::binary);
    f_size.open(size_file_path);
    return;
}

int PullStreamModule::process(Task &task) {
    if (f_size.eof()){
        task.fill_output_packet(0, Packet::generate_eof_packet());
        task.set_timestamp(DONE);
        return 0;
    }
    char buffer[50];
    Packet packet;
    f_size.getline(buffer, 20);
//    static int start_code = 0;
    if (pts_file_path_.empty()){
        pts_=pts_+66;
    }
    else{
        f_pts>>pts_;
    }

//    if (start_code==0)
//    {
//        start_code = pts_;
//    }
//    pts_=pts_-start_code;
    std::cout<<"pts:"<<pts_<<std::endl;
    std::string size_str(buffer);
    std::cout<<"size_str"<<size_str<<std::endl;
    int packet_size = std::stoi(size_str);
    std::cout << "video length:" << packet_size << std::endl;
    BMFAVPacket pkt = BMFAVPacket(NULL, packet_size);
    pkt.set_pts(pts_);
    if (packet_size>0) {
        f_data.read((char *) (pkt.get_data()), packet_size);
    }
//    pts_+=20;
    packet.set_data(pkt);
    packet.set_data_class_type(PACKET_TYPE);
    packet.set_class_name("libbmf_module_sdk.BMFAVPacket");
    packet.set_data_type(DATA_TYPE_C);
    timestamp_++;
    packet.set_timestamp(timestamp_);
    task.fill_output_packet(0, packet);

    return 0;
}

REGISTER_MODULE_CLASS(PullStreamModule)
