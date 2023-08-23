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
#include <bmf/sdk/packet.h>

namespace bmf_sdk {

PacketImpl::PacketImpl(void *obj, const TypeInfo *type_info,
                       const std::function<void(void *)> &del)
    : obj_(obj), type_info_(type_info), del_(del) {
    HMP_REQUIRE(obj_, "PacketImpl: null object detected");
    HMP_REQUIRE(type_info_, "PacketImpl: null type_info detected");
}

PacketImpl::~PacketImpl() {
    if (del_) {
        del_(obj_);
    }
}

PacketImpl *Packet::unsafe_self() { return self.get(); }

const PacketImpl *Packet::unsafe_self() const { return self.get(); }

const TypeInfo &Packet::type_info() const {
    HMP_REQUIRE(*this, "Packet: null packet");
    return self->type_info();
}

//
void Packet::set_timestamp(int64_t timestamp) {
    HMP_REQUIRE(*this, "Packet: null packet");
    self->set_timestamp(timestamp);
}
//
void Packet::set_time(double time) {
    HMP_REQUIRE(*this, "Packet: null packet");
    self->set_time(time);
}

int64_t Packet::timestamp() const {
    HMP_REQUIRE(*this, "Packet: null packet");
    return self->timestamp();
}

double Packet::time() const {
    HMP_REQUIRE(*this, "Packet: null packet");
    return self->time();
}

Packet Packet::generate_eos_packet() {
    Packet pkt = Packet(0);
    pkt.set_timestamp(EOS);
    return pkt;
}

Packet Packet::generate_eof_packet() {
    Packet pkt = Packet(0);
    pkt.set_timestamp(BMF_EOF);
    return pkt;
}

std::size_t string_hash(const char *str) {
    return std::hash<std::string_view>{}(str);
}

} // namespace bmf_sdk
