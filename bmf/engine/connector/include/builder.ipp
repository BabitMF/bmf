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
namespace bmf::builder {
    template<typename T,
            typename std::enable_if<std::is_integral<T>{} || std::is_floating_point<T>{} ||
                                    std::is_convertible<T, std::string const &>{} ||
                                    std::is_convertible<T, nlohmann::json>{}, bool>::type>
    Node Stream::FFMpegFilter(std::vector<Stream> inStreams, std::string const &filterName, T filterPara,
                              std::string const &alias) {
        bmf_sdk::JsonParam realPara;
        realPara.json_value_["name"] = filterName;
        realPara.json_value_["para"] = filterPara;
        return InternalFFMpegFilter(inStreams, filterName, realPara, alias);
    }

    template<typename T>
    Node Stream::Vflip(T para, std::string const &alias) {
        return FFMpegFilter({}, "vflip", para, alias);
    }

    template<typename T>
    Node Stream::Scale(T para, std::string const &alias) {
        return FFMpegFilter({}, "scale", para, alias);
    }

    template<typename T>
    Node Stream::Setsar(T para, std::string const &alias) {
        return FFMpegFilter({}, "setsar", para, alias);
    }

    template<typename T>
    Node Stream::Pad(T para, std::string const &alias) {
        return FFMpegFilter({}, "pad", para, alias);
    }

    template<typename T>
    Node Stream::Trim(T para, std::string const &alias) {
        return FFMpegFilter({}, "trim", para, alias);
    }

    template<typename T>
    Node Stream::Setpts(T para, std::string const &alias) {
        return FFMpegFilter({}, "setpts", para, alias);
    }

    template<typename T>
    Node Stream::Loop(T para, std::string const &alias) {
        return FFMpegFilter({}, "loop", para, alias);
    }

    template<typename T>
    Node Stream::Split(T para, std::string const &alias) {
        return FFMpegFilter({}, "split", para, alias);
    }

    template<typename T>
    Node Stream::Adelay(T para, std::string const &alias) {
        return FFMpegFilter({}, "adelay", para, alias);
    }

    template<typename T>
    Node Stream::Atrim(T para, std::string const &alias) {
        return FFMpegFilter({}, "atrim", para, alias);
    }

    template<typename T>
    Node Stream::Afade(T para, std::string const &alias) {
        return FFMpegFilter({}, "afade", para, alias);
    }

    template<typename T>
    Node Stream::Asetpts(T para, std::string const &alias) {
        return FFMpegFilter({}, "asetpts", para, alias);
    }

    template<typename T>
    Node Stream::Amix(std::vector<Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "amix", para, alias);
    }

    template<typename T>
    Node Stream::Overlay(std::vector<Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "overlay", para, alias);
    }

    template<typename T>
    Node Stream::Concat(std::vector<Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "concat", para, alias);
    }

    template<typename T,
            typename std::enable_if<std::is_integral<T>{} || std::is_floating_point<T>{} ||
                                    std::is_convertible<T, std::string const &>{} ||
                                    std::is_convertible<T, nlohmann::json>{}, bool>::type>
    Node Node::FFMpegFilter(std::vector<class Stream> inStreams, std::string const &filterName, T filterPara,
                            std::string const &alias) {
        bmf_sdk::JsonParam realPara;
        realPara.json_value_["name"] = filterName;
        realPara.json_value_["para"] = filterPara;
        return InternalFFMpegFilter(inStreams, filterName, realPara, alias);
    }

    template<typename T>
    Node Node::Vflip(T para, std::string const &alias) {
        return FFMpegFilter({}, "vflip", para, alias);
    }

    template<typename T>
    Node Node::Scale(T para, std::string const &alias) {
        return FFMpegFilter({}, "scale", para, alias);
    }

    template<typename T>
    Node Node::Setsar(T para, std::string const &alias) {
        return FFMpegFilter({}, "setsar", para, alias);
    }

    template<typename T>
    Node Node::Pad(T para, std::string const &alias) {
        return FFMpegFilter({}, "pad", para, alias);
    }

    template<typename T>
    Node Node::Trim(T para, std::string const &alias) {
        return FFMpegFilter({}, "trim", para, alias);
    }

    template<typename T>
    Node Node::Setpts(T para, std::string const &alias) {
        return FFMpegFilter({}, "setpts", para, alias);
    }

    template<typename T>
    Node Node::Loop(T para, std::string const &alias) {
        return FFMpegFilter({}, "loop", para, alias);
    }

    template<typename T>
    Node Node::Split(T para, std::string const &alias) {
        return FFMpegFilter({}, "split", para, alias);
    }

    template<typename T>
    Node Node::Adelay(T para, std::string const &alias) {
        return FFMpegFilter({}, "adelay", para, alias);
    }

    template<typename T>
    Node Node::Atrim(T para, std::string const &alias) {
        return FFMpegFilter({}, "atrim", para, alias);
    }

    template<typename T>
    Node Node::Afade(T para, std::string const &alias) {
        return FFMpegFilter({}, "afade", para, alias);
    }

    template<typename T>
    Node Node::Asetpts(T para, std::string const &alias) {
        return FFMpegFilter({}, "asetpts", para, alias);
    }

    template<typename T>
    Node Node::Amix(std::vector<class Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "amix", para, alias);
    }

    template<typename T>
    Node Node::Overlay(std::vector<class Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "overlay", para, alias);
    }

    template<typename T>
    Node Node::Concat(std::vector<class Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "concat", para, alias);
    }

    template<typename T,
            typename std::enable_if<std::is_integral<T>{} || std::is_floating_point<T>{} ||
                                    std::is_convertible<T, std::string const &>{} ||
                                    std::is_convertible<T, nlohmann::json>{}, bool>::type>
    Node Graph::FFMpegFilter(std::vector<Stream> inStreams, std::string const &filterName, T filterPara,
                             std::string const &alias) {
        bmf_sdk::JsonParam realPara;
        realPara.json_value_["name"] = filterName;
        realPara.json_value_["para"] = filterPara;
        return InternalFFMpegFilter(inStreams, filterName, realPara, alias);
    }

    template<typename T>
    Node Graph::Vflip(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "vflip", para, alias);
    }

    template<typename T>
    Node Graph::Scale(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "scale", para, alias);
    }

    template<typename T>
    Node Graph::Setsar(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "setsar", para, alias);
    }

    template<typename T>
    Node Graph::Pad(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "pad", para, alias);
    }

    template<typename T>
    Node Graph::Trim(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "trim", para, alias);
    }

    template<typename T>
    Node Graph::Setpts(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "setpts", para, alias);
    }

    template<typename T>
    Node Graph::Loop(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "loop", para, alias);
    }

    template<typename T>
    Node Graph::Split(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "split", para, alias);
    }

    template<typename T>
    Node Graph::Adelay(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "adelay", para, alias);
    }

    template<typename T>
    Node Graph::Atrim(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "atrim", para, alias);
    }

    template<typename T>
    Node Graph::Afade(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "afade", para, alias);
    }

    template<typename T>
    Node Graph::Asetpts(Stream inStream, T para, std::string const &alias) {
        return FFMpegFilter({inStream}, "asetpts", para, alias);
    }

    template<typename T>
    Node Graph::Amix(std::vector<Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "amix", para, alias);
    }

    template<typename T>
    Node Graph::Overlay(std::vector<Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "overlay", para, alias);
    }

    template<typename T>
    Node Graph::Concat(std::vector<Stream> inStreams, T para, std::string const &alias) {
        return FFMpegFilter(inStreams, "concat", para, alias);
    }
}
