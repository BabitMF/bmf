#include <filesystem>
#include <gtest/gtest.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/module_functor.h>

using namespace bmf_sdk;
namespace fs = std::filesystem;

namespace {


// two output streams
class FunctorSource : public Module
{
    int ivalue_ = 0;
    int close_time_ = 0;
    bool done_ = false;
    int npkts_[2] = {1, 1};
public:
    FunctorSource(int node_id, const JsonParam &option) 
        : Module(node_id, option)
    {
        if(option.json_value_.contains("ivalue")){
            ivalue_ = option.get<int>("ivalue");
        }

        if(option.json_value_.contains("npkts_0")){
            npkts_[0] = option.get<int>("npkts_0");
        }

        if(option.json_value_.contains("npkts_1")){
            npkts_[1] = option.get<int>("npkts_1");
        }
    }

    void mark_done()
    {
        done_ = true;
    }

    int process(Task &task) override
    {
        if(done_){
            task.set_timestamp(DONE);
        }

        for(int i = 0; i < 2; ++i){
            for(int j = 0; j < npkts_[i]; ++j){
                task.fill_output_packet(i, ivalue_ + i + j);
            }
        }

        ivalue_ += 2;
        return 0;
    }
    int close() override
    {
        close_time_++;
        HMP_REQUIRE(close_time_ < 2, "close module more than once. close time = {}", close_time_);
        return 0;
    }
};

// two input streams and one output stream
class FunctorAdd : public Module
{
    bool done_ = false;
public:
    using Module::Module;

    void mark_done()
    {
        done_ = true;
    }

    int process(Task &task) override
    {
        if(done_){
            task.set_timestamp(DONE);
        }

        Packet pkt0, pkt1;
        task.pop_packet_from_input_queue(0, pkt0);
        task.pop_packet_from_input_queue(1, pkt1);

        auto result = pkt0.get<int>() + pkt1.get<int>();

        task.fill_output_packet(0, Packet(result));

        return 0;
    }
};


REGISTER_MODULE_CLASS(FunctorSource)
REGISTER_MODULE_CLASS(FunctorAdd)

} //namespace



TEST(module_functor, source_add)
{
    JsonParam json;
    auto ivalue = 10;
    json.parse("{\"ivalue\": 10}");
    auto source = make_sync_func<std::tuple<>, std::tuple<int, int>>(
                         ModuleInfo("FunctorSource", "c++", ""),
                         json);
    auto add = make_sync_func<std::tuple<int, int>, std::tuple<int>>(
                         ModuleInfo("FunctorAdd", "c++", ""));

    for(int i = 0; i < 10; ++i){
        int a, b;
        std::tie(a, b) = source();
        EXPECT_EQ(a, ivalue);
        EXPECT_EQ(b, ivalue + 1);
        ivalue += 2;

        int c;
        std::tie(c) = add(a, b);
        EXPECT_EQ(c, a + b);
    }
};


TEST(module_functor, process_done)
{
    JsonParam json;
    auto ivalue = 10;
    json.parse("{\"ivalue\": 10}");
    auto source = make_sync_func<std::tuple<>, std::tuple<int, int>>(
                         ModuleInfo("FunctorSource", "c++", ""),
                         json);

    for(int i = 0; i < 10; ++i){
        int a, b;
        std::tie(a, b) = source();
        EXPECT_EQ(a, ivalue);
        EXPECT_EQ(b, ivalue + 1);
        ivalue += 2;
    }

    {
        dynamic_cast<FunctorSource &>(source.module()).mark_done();
        EXPECT_NO_THROW(source()); //
        EXPECT_THROW(source(), ProcessDone);
    }

};


TEST(module_functor, irregular_outputs)
{
    //irregular outputs
    {
        std::vector<std::tuple<int, int>> configs{
            {0, 0},
            {0, 2},
            {2, 0},
            {2, 2},
        };

        for (auto &config : configs)
        {
            JsonParam json;
            int n0, n1;
            std::tie(n0, n1) = config;
            json.parse(fmt::format("{{\"npkts_0\": {}, \"npkts_1\": {}}}", n0, n1));
            auto source = make_sync_func<std::tuple<>, std::tuple<int, int>>(
                ModuleInfo("FunctorSource", "c++", ""),
                json);

            source.execute();
            auto outs_0 = source.fetch<0>();
            auto outs_1 = source.fetch<1>();

            EXPECT_EQ(n0, outs_0.size());
            EXPECT_EQ(n1, outs_1.size());
        }
    }

    // multi-execute, single fetch
    {
        JsonParam json;
        json.parse(fmt::format("{{\"npkts_0\": {}, \"npkts_1\": {}}}", 1, 2));
        auto source = make_sync_func<std::tuple<>, std::tuple<int, int>>(
            ModuleInfo("FunctorSource", "c++", ""),
            json);

        // with cleanup
        source.execute(true);
        source.execute(true);

        auto outs_0 = source.fetch<0>();
        auto outs_1 = source.fetch<1>();

        EXPECT_EQ(1, outs_0.size());
        EXPECT_EQ(2, outs_1.size());

        // without cleanup
        source.execute(false);
        source.execute(false);

        outs_0 = source.fetch<0>();
        outs_1 = source.fetch<1>();

        EXPECT_EQ(2, outs_0.size());
        EXPECT_EQ(4, outs_1.size());
    }

}
