#include <bmf/sdk/bmf_av_packet.h>

namespace bmf_sdk{

struct BMFAVPacket::Private
{
    Private(const Tensor &d) : data(d) {}

	Tensor data;
};

BMFAVPacket::BMFAVPacket(const Tensor &data)
{
    HMP_REQUIRE(data.defined(), "BMFAVPacket: data is undefined");
    HMP_REQUIRE(data.device_type() == kCPU, 
        "BMFAVPacket: only support cpu buffer");
    HMP_REQUIRE(data.is_contiguous(), 
        "BMFAVPacket: only support contiguous buffer")
    self = std::make_shared<Private>(data);
}

BMFAVPacket::BMFAVPacket(int size, const TensorOptions &options)
    : BMFAVPacket(hmp::empty({size}, options))
{
}

BMFAVPacket::operator bool() const
{
    return self.get() != nullptr;
}


Tensor& BMFAVPacket::data()
{
    HMP_REQUIRE(*this, "BMFAVPacket: undefined BMFAVPacket detected");
    return self->data;
}

const Tensor& BMFAVPacket::data() const
{
    HMP_REQUIRE(*this, "BMFAVPacket: undefined BMFAVPacket detected");
    return self->data;
}


void *BMFAVPacket::data_ptr()
{
    return *this ? data().unsafe_data() : nullptr;
}

const void *BMFAVPacket::data_ptr() const
{
    return *this ? data().unsafe_data() : nullptr;
}

int BMFAVPacket::nbytes() const
{
    return *this ? data().nbytes() : 0;
}

BMFAVPacket& BMFAVPacket::copy_props(const BMFAVPacket &from)
{
    OpaqueDataSet::copy_props(from);
    SequenceData::copy_props(from);
    return *this;
}

int64_t BMFAVPacket::get_offset() const {
    return offset_;
}

void BMFAVPacket::set_offset(int64_t offset) {
    offset_ = offset;
}

int BMFAVPacket::get_whence() const {
    return whence_;
}

void BMFAVPacket::set_whence(int whence) {
    whence_ = whence;
}

} //namespace
