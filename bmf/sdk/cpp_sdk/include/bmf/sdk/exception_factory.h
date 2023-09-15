/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#ifndef MODULE_SDK_EXCEPTION_H
#define MODULE_SDK_EXCEPTION_H

#include <string>
#include <exception>
#include <bmf/sdk/common.h>

#include <bmf/sdk/exception_factory.h>
/** @ingroup CppMdSDK
    @brief Call the error handler.
    @param code one of Error::Code
    @param msg error message
    */
#define BMF_Error(code, msg)                                                   \
    bmf_sdk::error(code, msg, __FUNCTION__, __FILE__, __LINE__)

/** @ingroup CppMdSDK
    @brief Call the error handler.
    @param code one of Error::Code
    @param msg error message
    */
#define BMF_Error_(code, ...)                                                  \
    bmf_sdk::error(code, bmf_sdk::format(__VA_ARGS__).c_str(), __FUNCTION__,   \
                   __FILE__, __LINE__)

BEGIN_BMF_SDK_NS

BMF_API std::string format(const char *fmt, ...);

/** @ingroup CppMdSDK
 */
/*! @brief Class passed to an error.

This class encapsulates all or almost all necessary
information about the error happened in the program. The exception is
usually constructed and thrown implicitly via BMF_Error
@see error
 */
class BMF_API Exception : public std::exception {
  public:
    /*!
     Default constructor
     */
    Exception();

    /*!
     Full constructor. Normally the constructor is not called explicitly.
     Instead, the macros BMF_Error(), BMF_Error_() are used.
    */
    Exception(int _code, const char *_err, const char *_func, const char *_file,
              int _line);

    virtual ~Exception() throw();

    /*!
     \return the error description and the context as a text string.
    */
    virtual const char *what() const throw();

    void formatMessage();

    std::string msg; ///< the formatted error message

    int code;         ///< error code @see BMFStatus
    std::string err;  ///< error description
    std::string func; ///< function name. Available only when the compiler
                      /// supports getting it
    std::string file; ///< source file name where the error has occurred
    int line; ///< line number in the source file where the error has occurred
};

BMF_API void error(int _code, const char *_err, const char *_func,
                   const char *_file, int _line);

END_BMF_SDK_NS
#endif // MODULE_SDK_EXCEPTION_H
