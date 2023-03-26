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

#ifndef BMF_SAFE_QUEUE_H
#define BMF_SAFE_QUEUE_H

#include <queue>
#include <list>
#include <mutex>
#include <thread>
#include <cstdint>
#include <condition_variable>
#include <bmf/sdk/common.h>
#include <bmf/sdk/trace.h>

BEGIN_BMF_ENGINE_NS
    USE_BMF_SDK_NS

/** A thread-safe asynchronous queue */
    template<class T>
    class SafeQueue {
    public:

        /*! Create safe queue. */
        SafeQueue() = default;

        SafeQueue(SafeQueue &&sq) {
            m_queue = std::move(sq.m_queue);
        }

        SafeQueue(std::shared_ptr<std::queue<T> > &queue) {
            m_queue = *queue.get();
        }

        SafeQueue(const SafeQueue &sq) {
            std::lock_guard<std::mutex> lock(sq.m_mutex);
            m_queue = sq.m_queue;
        }

        /*! Destroy safe queue. */
        ~SafeQueue() {
            std::lock_guard<std::mutex> lock(m_mutex);
        }

        /**
         * Sets the maximum number of items in the queue. Defaults is 0: No limit
         * \param[in] item An item.
         */
        void set_max_num_items(unsigned int max_num_items) {
            m_max_num_items = max_num_items;
        }

        /**
         * @brief Set the identifier based on the stream
         * \param[in] identifier Identifier string.
         */
        void set_identifier(std::string const &identifier) {
            identifier_ = identifier;
        }

        /**
         *  Pushes the item into the queue.
         * \param[in] item An item.
         * \return true if an item was pushed into the queue
         */
        bool push(const T &item) {
            std::lock_guard<std::mutex> lock(m_mutex);

            if (m_max_num_items > 0 && m_queue.size() > m_max_num_items)
                return false;

            m_queue.push(item);
            BMF_TRACE_QUEUE_INFO(identifier_.c_str(), m_queue.size(), m_max_num_items);
            return true;
        }

        /**
         *  Pushes the item into the queue.
         * \param[in] item An item.
         * \return true if an item was pushed into the queue
         */
        bool push(const T &&item) {
            std::lock_guard<std::mutex> lock(m_mutex);

            if (m_max_num_items > 0 && m_queue.size() > m_max_num_items)
                return false;

            m_queue.push(item);
            BMF_TRACE_QUEUE_INFO(identifier_.c_str(), m_queue.size(), m_max_num_items);
            return true;
        }

        bool front(T &item) {
            std::unique_lock<std::mutex> lock(m_mutex);

            if (m_queue.empty())
                return false;
            item = m_queue.front();

            return true;
        }

        /**
         *   pop item from the queue.
         * \param[out] item The item.
         * \return False is returned if no item is available.
         */
        bool pop(T &item) {
            std::unique_lock<std::mutex> lock(m_mutex);

            if (m_queue.empty())
                return false;

            item = m_queue.front();
            m_queue.pop();
            BMF_TRACE_QUEUE_INFO(identifier_.c_str(), m_queue.size(), m_max_num_items);
            return true;
        }


        /**
         *  Gets the number of items in the queue.
         * \return Number of items in the queue.
         */
        size_t size() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.size();
        }

        /**
         *  Check if the queue is empty.
         * \return true if queue is empty.
         */
        bool empty() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.empty();
        }

        /*! The copy assignment operator */
        SafeQueue &operator=(const SafeQueue &sq) {
            if (this != &sq) {
                std::lock_guard<std::mutex> lock1(m_mutex);
                std::lock_guard<std::mutex> lock2(sq.m_mutex);
                std::queue<T> temp{sq.m_queue};
                m_queue.swap(temp);
            }

            return *this;
        }

        /*! The move assignment operator */
        SafeQueue &operator=(SafeQueue &&sq) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue = std::move(sq.m_queue);

            return *this;
        }


    private:

        std::queue<T> m_queue;
        mutable std::mutex m_mutex;
        unsigned int m_max_num_items = 0;
        std::string identifier_;
    };


    template<class T>
    class SafePriorityQueue {
    public:

        /*! Create safe queue. */
        SafePriorityQueue() = default;

        SafePriorityQueue(SafePriorityQueue &&sq) {
            m_queue = std::move(sq.m_queue);
        }

        SafePriorityQueue(const SafePriorityQueue &sq) {
            std::lock_guard<std::mutex> lock(sq.m_mutex);
            m_queue = sq.m_queue;
        }

        /*! Destroy safe queue. */
        ~SafePriorityQueue() {
            std::lock_guard<std::mutex> lock(m_mutex);
        }

        /**
         * Sets the maximum number of items in the queue. Defaults is 0: No limit
         * \param[in] item An item.
         */
        void set_max_num_items(unsigned int max_num_items) {
            m_max_num_items = max_num_items;
        }

        /**
         *  Pushes the item into the queue.
         * \param[in] item An item.
         * \return true if an item was pushed into the queue
         */
        bool push(const T &item) {
            std::lock_guard<std::mutex> lock(m_mutex);

            if (m_max_num_items > 0 && m_queue.size() > m_max_num_items)
                return false;

            m_queue.push(item);
            return true;
        }

        /**
         *  Pushes the item into the queue.
         * \param[in] item An item.
         * \return true if an item was pushed into the queue
         */
        bool push(const T &&item) {
            std::lock_guard<std::mutex> lock(m_mutex);

            if (m_max_num_items > 0 && m_queue.size() > m_max_num_items)
                return false;

            m_queue.push(item);
            return true;
        }

        /**
         *   pop item from the queue.
         * \param[out] item The item.
         * \return False is returned if no item is available.
         */
        bool pop(T &item) {
            std::unique_lock<std::mutex> lock(m_mutex);

            if (m_queue.empty())
                return false;

            item = m_queue.top();
            m_queue.pop();
            return true;
        }


        /**
         *  Gets the number of items in the queue.
         * \return Number of items in the queue.
         */
        size_t size() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.size();
        }

        /**
         *  Check if the queue is empty.
         * \return true if queue is empty.
         */
        bool empty() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.empty();
        }

        /*! The copy assignment operator */
        SafePriorityQueue &operator=(const SafePriorityQueue &sq) {
            if (this != &sq) {
                std::lock_guard<std::mutex> lock1(m_mutex);
                std::lock_guard<std::mutex> lock2(sq.m_mutex);
                std::queue<T> temp{sq.m_queue};
                m_queue.swap(temp);
            }

            return *this;
        }

        /*! The move assignment operator */
        SafePriorityQueue &operator=(SafePriorityQueue &&sq) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue = std::move(sq.m_queue);

            return *this;
        }


    private:

        std::priority_queue<T, std::vector<T>> m_queue;
        mutable std::mutex m_mutex;
        unsigned int m_max_num_items = 0;
    };
END_BMF_ENGINE_NS
#endif //BMF_SAFE_QUEUE_H
