#include "normal_image_info.h"
#include <boost/algorithm/string.hpp>
using namespace boost::python;
class Rect{
    public:
        Rect(int x,int y,int width,int height){
        this->x=x;
        this->y=y;
        this->width=width;
        this->height = height;
        };
        Rect(){
        x=0;
        y=0;
        width = 0;
        height = 0;
        };
        int x,y,width,height;
};

void NormalImageInfo::process(Task &task) {
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
            const boost::python::dict &dict_data = pkt.get_data<boost::python::dict>();
            // Create output frame
            Rect rect=Rect();
            int width = 0;
            int height = 0;
            std::string image_info ="";
            get_dict_value(dict_data,"rect.x",rect.x);
            get_dict_value(dict_data,"rect.y",rect.y);
            get_dict_value(dict_data,"rect.width",rect.width);
            get_dict_value(dict_data,"rect.height",rect.height);
            get_dict_value(dict_data,"width",width);
            get_dict_value(dict_data,"height",height);
            get_dict_value(dict_data,"image_info",image_info);
            dict result_dict =dict();
            result_dict.update(dict_data);
            result_dict["width"]=width+1000;
            result_dict["image_info"]=image_info+" c_module";
            // Add output frame to output queue
            Packet output_pkt;
            output_pkt.set_timestamp(pkt.get_timestamp());
            output_pkt.set_data<dict>(result_dict);
            task.add_packet_to_out_queue(label, output_pkt);
        }
    }
}

BMF_MODULE(normal_image_info, NormalImageInfo)
