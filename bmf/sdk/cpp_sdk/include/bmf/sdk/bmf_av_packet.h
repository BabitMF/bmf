#pragma once

#include <bmf/sdk/hmp_import.h>
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/bmf_type_info.h>

namespace bmf_sdk{

class BMF_API BMFAVPacket : public OpaqueDataSet,
                            public SequenceData
{
    struct Private;

    std::shared_ptr<Private> self;
public:
    BMFAVPacket() = default;

    BMFAVPacket(const BMFAVPacket&) = default;

    BMFAVPacket(BMFAVPacket&&) = default;

    BMFAVPacket& operator=(const BMFAVPacket&) = default;

    /**
     * @brief Construct a new BMFAVPacket object
     * 
     * @param data contiguous tensor data, cpu only
     */
    BMFAVPacket(const Tensor &data);

    /**
     * @brief Construct a new BMFAVPacket object
     * 
     * @param size 
     * @param options ref VideoFrame
     */
    BMFAVPacket(int size, const TensorOptions &options = kUInt8);

    /**
     * @brief 
     * 
     * @tparam Options 
     * @param size 
     * @param opts ref VideoFrame
     * @return BMFAVPacket 
     */
    template <typename... Options>
    static BMFAVPacket make(int size, Options &&...opts)
    {
        return BMFAVPacket(size,
            TensorOptions(kUInt8).options(std::forward<Options>(opts)...));
    }

    /**
     * @brief check if BMFAVPacket if defined
     * 
     * @return true 
     * @return false 
     */
    operator bool() const;

    /**
     * @brief 
     * 
     * @return Tensor& 
     */
    Tensor& data();
    
    /**
     * @brief 
     * 
     * @return const Tensor& 
     */
    const Tensor& data() const;

    /**
     * @brief return raw pointer of underlying data
     * 
     * @return void* 
     */
    void *data_ptr();
    
    /**
     * @brief 
     * 
     * @return const void* 
     */
    const void *data_ptr() const;

    /**
     * @brief number of bytes of underlying data
     * 
     * @return int 
     */
    int nbytes() const;


    /**
     * @brief copy all extra props(set by member func set_xxx) from `from`(deepcopy if needed), 
     * 
     * @param from 
     * @return VideoFrame& 
     */
    BMFAVPacket& copy_props(const BMFAVPacket &from);
};


} //namespace bmf_sdk


BMF_DEFINE_TYPE(bmf_sdk::BMFAVPacket)