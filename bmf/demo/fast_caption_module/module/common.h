/*
    Copyright 2024 Babit Authors
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/
#ifndef _COMMON_H
#define _COMMON_H

inline void set_string(std::string key, std::string &val, nlohmann::json &json_param)
{
    if (json_param.find(key) != json_param.end() && json_param[key].is_string())
    {
        val = json_param[key];
    }
}

template <typename T>
inline void set_number(std::string key, T &val, nlohmann::json &json_param)
{
    if (json_param.find(key) != json_param.end() && json_param[key].is_number())
    {
        val = (float)json_param[key];
    }
}

template <typename T>
void load(const std::string &path, std::vector<T> &host)
{
    std::fstream file(path, std::ios::binary | std::ios::in);
    if (file.is_open())
    {
        auto rd_buf = file.rdbuf();
        size_t size = rd_buf->pubseekoff(std::ios::beg, std::ios::end, std::ios::in);
        rd_buf->pubseekpos(std::ios::beg, std::ios::in);
        rd_buf->sgetn((char *)host.data(), std::min(size, host.size() * sizeof(T)));
        file.close();
    }
    else
    {
        LOG(ERROR) << "file does not exists: " << path;
    }
}

template <typename T>
void write(const std::string &path, at::Tensor &data)
{
    auto sizes = data.sizes();
    int32_t s = 1;
    for (size_t i = 0; i < sizes.size(); i++)
    {
        s *= sizes.data()[i];
    }
    std::vector<T> host(s);
    cudaMemcpy(host.data(), data.data_ptr(), s * sizeof(T), cudaMemcpyDefault);
    std::fstream file(path, std::ios::binary | std::ios::out);
    if (file.is_open())
    {
        file.write((const char *)host.data(), host.size() * sizeof(T));
        file.close();
    }
}

template <typename T>
void write(const std::string &path, std::vector<T> &host)
{
    std::fstream file(path, std::ios::binary | std::ios::out);
    if (file.is_open())
    {
        file.write((const char *)host.data(), host.size() * sizeof(T));
        file.close();
    }
}

template <class Data>
class cpu_executor
{
    using pack_t = std::packaged_task<Data(Data)>;

public:
    static cpu_executor *instance()
    {
        static cpu_executor exec;
        return &exec;
    }

    ~cpu_executor()
    {
        working_ = false;
        for (auto &worker : workers_)
        {
            if (worker.joinable())
            {
                work_cv_.notify_all();
                worker.join();
            }
        }
    }

    void set_worker(int32_t num_worker)
    {
        for (int32_t i = workers_.size(); i < num_worker; ++i)
        {
            workers_.emplace_back(std::thread([&]()
                                              { working(i); }));
        }
    }

    std::future<Data> exec(std::function<Data(Data)> f, Data task)
    {
        pack_t pack(f);
        std::future<Data> future = pack.get_future();
        auto data_pack = std::make_pair<pack_t, Data>(std::forward<pack_t>(std::move(pack)),
                                                      std::forward<Data>(std::move(task)));

        std::lock_guard<std::mutex> guard(work_mutex_);
        work_queue_.emplace(std::move(data_pack));
        work_cv_.notify_one();
        return future;
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_cv_.wait(lock, [&]() -> bool
                      { return work_queue_.empty(); });
    }

private:
    cpu_executor()
    {
        for (int32_t i = 0; i < 1; i++)
        {
            workers_.emplace_back(std::thread([&, i]()
                                              { working(i); }));
        }
    }

    void working(int32_t thread_idx)
    {
        char name[128] = {0};
        sprintf(name, "cpu_exec_%02d", thread_idx);
        prctl(PR_SET_NAME, name);
        while (working_)
        {
            auto worker_size = workers_.size();
            if (thread_idx == 0 && work_queue_.size() > 1 && worker_size < 12)
            { // TODO : worker_size limit be variable
                LOG(INFO) << "before work worker queue size is " << work_queue_.size();
                workers_.emplace_back(std::thread([&, worker_size]()
                                                  { working(worker_size); }));
                LOG(INFO) << "auto create worker thread " << worker_size
                          << ". workqueue size is " << work_queue_.size()
                          << ".";
            }
            std::unique_lock<std::mutex> lock(work_mutex_);
            work_cv_.wait(lock, [&]() -> bool
                          { return !work_queue_.empty() || !working_; });
            if (working_)
            {
                auto data_pack = std::move(work_queue_.front());
                work_queue_.pop();
                lock.unlock();
                data_pack.first(std::move(data_pack.second));
            }
            else
            {
                lock.unlock();
            }
            work_cv_.notify_one();
        }
    }

    volatile bool working_ = true;
    std::vector<std::thread> workers_{};
    std::condition_variable work_cv_{};
    std::queue<std::pair<pack_t, Data>> work_queue_{};
    std::mutex work_mutex_{};
};

#endif