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
#include <bmf/sdk/exception_factory.h>
#include <bmf/sdk/error_define.h>

#include <cstdarg>

BEGIN_BMF_SDK_NS
    std::string format(const char *fmt, ...) {
        va_list args;
        char buff_size[1024];
        va_start(args, fmt);
        vsnprintf(buff_size, 1023, fmt, args);
        va_end(args);
        std::string result = buff_size;
        return result;
    }

    Exception::Exception() {
        code = 0;
        line = 0;
    }

    Exception::Exception(int _code, const char *_err, const char *_func, const char *_file,
                         int _line)
            : code(_code), err(_err), func(_func), file(_file), line(_line) {
        formatMessage();
    }

    Exception::~Exception() throw() {}

/*!
 \return the error description and the context as a text string.
 */
    const char *Exception::what() const throw() { return msg.c_str(); }

    void Exception::formatMessage() {
        msg = format("BMF(%s) %s:%d: error: (%d:%s) %s in function '%s'\n", BMF_SDK_VERSION, file.c_str(), line,
                     code, BMFErrorStr(code), err.c_str(), func.c_str());
    }

    void error(int _code, const char *_err, const char *_func, const char *_file, int _line) {
        Exception exception = Exception(_code, _err, _func, _file, _line);
        throw exception;
    }

END_BMF_SDK_NS