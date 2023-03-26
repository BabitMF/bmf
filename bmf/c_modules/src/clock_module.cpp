#include "../include/clock_module.h"


#include <unistd.h>

#include <string>

ClockModule::ClockModule(int node_id, JsonParam option) {
    int32_t fps = 25;
    // 1 second
    tick_ = std::chrono::high_resolution_clock::duration(1000000000);
    time_base_ = Fraction::Fraction(1, 1000000);
    frm_cnt_ = 0;

    if (option.has_key("fps")) {
        option.get_int("fps", fps);
        if (fps <= 0)
            throw std::logic_error("Wrong fps provided.");
    }
    if (option.has_key("time_base")) {
        std::string tb;
        option.get_string("time_base", tb);
        time_base_ = Fraction::Fraction(tb);
        if (time_base_.neg())
            throw std::logic_error("Wrong time_base provided.");
    }

    tick_ /= fps;
    fps_tick_ = Fraction::Fraction(1, fps);
}

int ClockModule::process(Task &task) {
    auto now = std::chrono::high_resolution_clock::now();

    if (lst_ts_.time_since_epoch() == std::chrono::high_resolution_clock::duration::zero()) {
        lst_ts_ = now;
    } else if (now - lst_ts_ < tick_) {
        // Can be 10us quicker, in order to decrease delay.
        auto sleep_time = (tick_ - (now - lst_ts_)).count() / 1000 - 10;
        if (sleep_time > 0)
            usleep(sleep_time);
    }

    lst_ts_ += tick_;
    Packet pkt(0);
    pkt.set_timestamp((fps_tick_ * frm_cnt_++).to_int_based(time_base_) * 1000000);
    //pkt.set_timestamp((fps_tick_ * frm_cnt_++).to_int_based(time_base_));
    task.fill_output_packet(0, pkt);
    return 0;
}

bool ClockModule::is_hungry(int input_stream_id) {
    return true;
}

REGISTER_MODULE_CLASS(ClockModule)
