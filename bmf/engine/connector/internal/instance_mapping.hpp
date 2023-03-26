#ifndef CONNECTOR_INSTANCE_MAPPING_HPP
#define CONNECTOR_INSTANCE_MAPPING_HPP

#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace bmf::internal {
        template<typename T>
        class InstanceMapping {
        public:
            InstanceMapping();

            uint32_t insert(std::shared_ptr<T> instance) {
                std::lock_guard<std::mutex> _(lk_);
                ++counting_;
                mapping_[counting_] = instance;
                ref_cnt_[counting_] = 1;
                return counting_;
            }

            bool ref(uint32_t idx) {
                std::lock_guard<std::mutex> _(lk_);
                if (mapping_.count(idx) == 0)
                    return false;
                ++ref_cnt_[idx];
                return true;
            }

            std::shared_ptr<T> get(uint32_t idx) {
                std::lock_guard<std::mutex> _(lk_);
                if (mapping_.count(idx) == 0)
                    throw std::range_error("Instance not existed.");
                return mapping_[idx];
            }

            bool exist(uint32_t idx) {
                std::lock_guard<std::mutex> _(lk_);
                return mapping_.count(idx) != 0;
            }

            bool remove(uint32_t idx) {
                std::lock_guard<std::mutex> _(lk_);
                if (mapping_.count(idx) == 0)
                    return false;
                --ref_cnt_[idx];
                if (ref_cnt_[idx] == 0) {
                    mapping_.erase(idx);
                    ref_cnt_.erase(idx);
                }
                return true;
            }

        private:
            uint32_t counting_;
            std::mutex lk_;
            std::map<uint32_t, std::shared_ptr<T> > mapping_;
            std::map<uint32_t, uint32_t> ref_cnt_;
        };

        template<typename T>
        InstanceMapping<T>::InstanceMapping() :counting_(0) {}
    }

#endif //CONNECTOR_INSTANCE_MAPPING_HPP
