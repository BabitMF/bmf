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

#ifndef BMF_TASK_H
#define BMF_TASK_H

#include <queue>
#include <map>
#include <bmf/sdk/common.h>
#include <bmf/sdk/packet.h>

BEGIN_BMF_SDK_NS
typedef std::map<int, std::shared_ptr<std::queue<Packet>>> PacketQueueMap;

/** @ingroup CppMdSDK
 */
class BMF_API Task {
  public:
    /*!
     * @brief construct Task.
     * @param node_id The id of the running task.
     * @param input_stream_id_list input stream id list.
     * @param output_stream_id_list output stream id list.
     */
    Task(int node_id = -1, std::vector<int> input_stream_id_list = {},
         std::vector<int> output_stream_id_list = {});

    Task(const Task &rhs);

    Task(Task &&rhs);

    Task &operator=(Task rhs);

    friend BMF_API void swap(Task &target, Task &source);

    /* @} */
    /*!
     * @brief fill packet into the input stream queue.
     * @param stream_id The id of the input stream.
     * @param packet the packet add to the input stream queue.
     * @return  true if success, false if failed.
     */
    bool fill_input_packet(int stream_id, Packet packet);

    /*!
     * @brief fill packet into the output stream queue.
     * @param stream_id The id of the output stream.
     * @param packet the packet add to the output stream queue.
     * @return  true if success, false if failed.
     */
    bool fill_output_packet(int stream_id, Packet packet);

    /*!
     * @brief pop packet from the given stream id of output queue.
     * @param stream_id The id of the output stream.
     * @param packet the packet poped from the output stream queue.
     * @return  true if success, false if failed.
     */
    bool pop_packet_from_out_queue(int stream_id, Packet &packet);

    /*!
     * @brief pop packet from the given stream id of input queue.
     * @param stream_id The id of the input stream.
     * @param packet the packet poped from the input stream queue.
     * @return  true if success, false if failed.
     */
    bool pop_packet_from_input_queue(int stream_id, Packet &packet);

    bool input_queue_empty(int stream_id);

    bool output_queue_empty(int stream_id);

    /*!
     * @brief get the timestamp of the task
     * @return  timestamp.
     */
    int64_t timestamp() const;

    /*!
     * @brief set the timestamp of the task.
     * @param t the timestamp of the task.
     * @return  true if success, false if failed.
     */
    void set_timestamp(int64_t t);

    /*!
     * @brief get output stream queue.
     * @return  output stream map.
     */
    PacketQueueMap &get_outputs();

    /*!
     * @brief get input stream queue.
     * @return  input stream map.
     */
    PacketQueueMap &get_inputs();

    /*!
     * @brief get input stream id list.
     * @return  input stream id list.
     */
    std::vector<int> get_input_stream_ids();

    /*!
     * @brief get output stream id list.
     * @return  output stream id list.
     */
    std::vector<int> get_output_stream_ids();

    int get_node();

    void init(int node_id, std::vector<int> input_stream_id_list,
              std::vector<int> output_stream_id_list);

  public:
    int64_t timestamp_ = UNSET;
    int node_id_;
    PacketQueueMap inputs_queue_;
    PacketQueueMap outputs_queue_;
};

END_BMF_SDK_NS
#endif // BMF_TASK_H
