#ifndef BMF_KAFKA_REPORTER_INTERFACE_H
#define BMF_KAFKA_REPORTER_INTERFACE_H
#include <iostream>
class BMFKafkaReporterI {
public:
    virtual ~BMFKafkaReporterI() = default;

    virtual bool produce(const std::string& message) = 0;
    virtual void flush() = 0;
    virtual int64_t get_num_produced() const = 0;
};
#endif