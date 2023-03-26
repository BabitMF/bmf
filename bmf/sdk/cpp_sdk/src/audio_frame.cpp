#include <bmf/sdk/audio_frame.h>
#include <hmp/format.h>

namespace bmf_sdk{


//copy from ffmpeg
static int popcount_c(uint32_t x)
{
    x -= (x >> 1) & 0x55555555;
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x += x >> 8;
    return (x + (x >> 16)) & 0x3F;
}


static inline int popcount64_c(uint64_t x)
{
    return popcount_c(uint32_t(x)) + popcount_c((uint32_t(x>>32)));
}


struct AudioFrame::Private
{
    Private(const TensorList &data_, bool planer_, uint64_t layout_)
        : data(data_), layout(layout_), planer(planer_)
    {
    }

    static SizeArray infer_shape(int samples, bool planer, uint64_t layout)
    {
        HMP_REQUIRE(layout != 0, "can not infer_shape when layout == 0");
        auto channels = popcount64_c(layout);
        if(planer){
            return SizeArray{channels, samples};
        }
        else{
            return SizeArray{samples, channels};
        }
    }

    static TensorList construct(int samples, uint64_t layout, bool planer, const TensorOptions &options)
    {
        Tensor data_s = hmp::empty(infer_shape(samples, planer, layout), options);
        auto channels = popcount64_c(layout);

        TensorList data;
        if(planer){
            for(int i = 0; i < channels; ++i){
                data.push_back(data_s.select(0, i));
            }
        }
        else{
            data = {data_s};
        }
        return data;
    }


	TensorList data;
    bool planer;
    uint64_t layout;

    float sample_rate = 1;
};


AudioFrame::AudioFrame(const TensorList &data, uint64_t layout, bool planer)
{
    //Note: channels from layout is not reliable if layout == 0
    auto channels = popcount64_c(layout);
    if(planer){
        HMP_REQUIRE(channels == 0 || data.size() == channels,
             "AudioFrame: data shape does not match channel layout, expect channels {}, got {}",
             channels, data.size());
        for(auto &d : data){
            HMP_REQUIRE(d.defined() && d.dim() ==1, 
                "AudioFrame: expect 1d data for planer audio frame");
            HMP_REQUIRE(d.device_type() == kCPU,
                "AudioFrame: only support cpu data");
        }
    }
    else{
        HMP_REQUIRE(data.size() == 1 && data[0].dim() ==2, 
            "AudioFrame: expect 2d data for interleave audio frame");
        HMP_REQUIRE(data[0].device_type() == kCPU,
            "AudioFrame: only support cpu data");

        HMP_REQUIRE(channels == 0 || data[0].size(1) == channels,
             "AudioFrame: data shape does not match channel layout, expect channels {}, got {}",
             channels, data[0].size(1));
    }

    self = std::make_shared<Private>(data, planer, layout);
}

AudioFrame::AudioFrame(int samples, uint64_t layout, bool planer, const TensorOptions &options)
    : AudioFrame(Private::construct(samples, layout, planer, options), layout, planer)
{
}


AudioFrame AudioFrame::clone() const
{
    if(*this){
        TensorList data;
        for(auto &d : self->data){
            data.push_back(d.clone());
        }

        auto copy = AudioFrame(data, self->layout, self->planer);
        copy.copy_props(*this);
        return copy;
    }
    else{
        return AudioFrame();
    }
}

AudioFrame::operator bool() const
{
    return self.get() != nullptr;
}


uint64_t AudioFrame::layout() const
{
    return self->layout;
}


bool AudioFrame::planer() const
{
    return self->planer;
}


int AudioFrame::nsamples() const
{
    return self->data[0].size(0);
}

int AudioFrame::nchannels() const
{
    if(self->planer){
        return self->data.size();
    }
    else{
        return self->data[0].size(1);
    }
}


ScalarType AudioFrame::dtype() const
{
    return self->data[0].dtype();
}

void AudioFrame::set_sample_rate(float sample_rate)
{
    HMP_REQUIRE(sample_rate > 0, 
        "AudioFrame: expect sample_rate > 0, got {}", sample_rate);
    self->sample_rate = sample_rate;
}


float AudioFrame::sample_rate() const
{
    return self->sample_rate;
}


const TensorList& AudioFrame::planes() const
{
    return self->data;
}

int AudioFrame::nplanes() const
{
    return self->data.size();
}

Tensor AudioFrame::plane(int p) const
{
    HMP_REQUIRE(self.get() && p < self->data.size(),
        "AudioFrame: plane index {} is out of range", p);

    return self->data[p];
}

Tensor AudioFrame::operator[](int p) const
{
    return plane(p);
}


AudioFrame& AudioFrame::copy_props(const AudioFrame &from)
{
    OpaqueDataSet::copy_props(from);
    SequenceData::copy_props(from);
    set_sample_rate(from.sample_rate());

    return *this;
}


} //namespace bmf_sdk