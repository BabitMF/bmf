#pragma once

#include <stdint.h>
#include <type_traits>
#include <typeinfo>
#include <bmf/sdk/common.h>

namespace bmf_sdk{


//to avoid design a sophistic type system, like boost::type_index,
//we provide template interface for user to register their types

struct TypeInfo
{
    const char *name;
    std::size_t index; //unique
};

static inline bool operator==(const TypeInfo& lhs, const TypeInfo &rhs)
{
    return lhs.index == rhs.index;
}


static inline bool operator!=(const TypeInfo& lhs, const TypeInfo &rhs)
{
    return lhs.index != rhs.index;
}

BMF_API std::size_t string_hash(const char *str);

//
template<typename T>
struct TypeTraits
{
    static const char *name()
    {
        return typeid(T).name();
    }
};


template<typename T>
const TypeInfo& _type_info()
{
    static TypeInfo s_type_info{TypeTraits<T>::name(), 
            string_hash(TypeTraits<T>::name())};
    return s_type_info;
}

template<typename T>
const TypeInfo& type_info()
{
    using U = std::remove_reference_t<std::remove_cv_t<T>>;
    return _type_info<U>();
}


//Note: use it in global namespace
#define BMF_DEFINE_TYPE_N(T, Name) \
    namespace bmf_sdk{ \
    template<> struct TypeTraits<T>{ \
        static const char *name() { return Name; } \
    }; }

#define BMF_DEFINE_TYPE(T) BMF_DEFINE_TYPE_N(T, #T)



} //namespace bmf_sdk

//std types
//BMF_DEFINE_TYPE(std::string)