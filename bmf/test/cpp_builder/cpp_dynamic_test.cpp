#include "builder.hpp"
#include <unistd.h>
#include "nlohmann/json.hpp"
#include "cpp_test_helper.h"

// Dynamic reset function test
TEST(cpp_dynamic_reset, reset_pass_through_node) {
    const std::string output_file = "./output_reset_cpp.mp4";
    const std::string input_file = "../../files/big_bunny_10s_30fps.mp4";
    BMF_CPP_FILE_REMOVE(output_file); 

    // 1. Create main graph
    auto main_graph = bmf::builder::Graph(bmf::builder::NormalMode);
    BMFLOG(BMF_INFO) << "Main graph created.";

    // 2. Add decoder node
    nlohmann::json decode_para = {
        {"input_path", input_file}, 
        {"alias", "decoder0"}     
    };
    auto decoder_node = main_graph.Decode(bmf_sdk::JsonParam(decode_para)); 
    auto video_stream = decoder_node["video"];  
    auto audio_stream = decoder_node["audio"];
    BMFLOG(BMF_INFO) << "Decoder node created successfully.";

    // 3. Add PassThrough node to be reset
    std::vector<bmf::builder::Stream> pass_through_inputs = {video_stream, audio_stream};
    nlohmann::json pass_through_para = {}; 
    bmf_sdk::JsonParam pass_through_option(nlohmann::json::object());
    const std::string python_module_dir = "../../../bmf/test/dynamical_graph";   
    auto pass_through_node = main_graph.Module(
        pass_through_inputs, 
        "reset_pass_through",                     
        bmf::builder::ModuleType::Python,      
        pass_through_option,  
        "reset_pass_through",  
        python_module_dir,             
        "",                                                     
        bmf::builder::InputManagerType::Immediate, 
        0                                  
    );
    BMFLOG(BMF_INFO) << "PassThrough node created successfully.";

    // 4. Non-blocking start graph
    main_graph.Start(true, true);
    BMFLOG(BMF_INFO) << "Waiting 20ms to ensure node initialization";
    usleep(20000);
    
    // 5. Construct dynamic reset configuration
    nlohmann::json reset_config = {
        {"alias", "reset_pass_through"}, 
        {"output_path", output_file},     
        {"video_params", {               
            {"codec", "h264"},
            {"width", 320},
            {"height", 240},
            {"crf", 23},
            {"preset", "veryfast"}
        }}
    };
    bmf_sdk::JsonParam reset_config_param(reset_config);
    BMFLOG(BMF_INFO) << "Dynamic reset configuration:\n" << reset_config.dump(2);

    // 6. Create empty reset graph
    auto temp_graph = bmf::builder::Graph(bmf::builder::NormalMode);

    // 7. Temporary graph describes reset information
    temp_graph.DynamicReset(reset_config_param);

    // 8. Main graph performs update
    int update_ret = main_graph.Update(temp_graph);
    if (update_ret != 0) {
        BMFLOG(BMF_ERROR) << "Dynamic reset call failed, return code: " << update_ret;
        FAIL() << "Dynamic reset node call failed.";
    }
    BMFLOG(BMF_INFO) << "Waiting 1 second to ensure processing is complete.";
    sleep(1); 

    // 9. Close graph
    int close_ret = main_graph.Close();
    if (close_ret != 0) {
        BMFLOG(BMF_ERROR) << "Graph close failed, return code: " << close_ret;
        FAIL() << "Graph close failed";
    }
    BMFLOG(BMF_INFO) << "Main graph closed successfully.";
}
