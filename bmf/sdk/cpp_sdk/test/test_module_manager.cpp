#include <gtest/gtest.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/module_manager.h>
#include <bmf/sdk/compat/path.h>

using namespace bmf_sdk;
namespace {

class InAppModuleDemo : public Module {
  public:
    using Module::Module;

    int32_t process(Task &task) override { return 0; }
};

REGISTER_MODULE_CLASS(InAppModuleDemo)
};

TEST(module_manager, test_compat_path) {
    auto p0 = fs::path("/home/foo");
    ASSERT_EQ(p0.string(), "/home/foo");
    ASSERT_EQ(std::string(p0), "/home/foo");

    auto p1 = p0 / std::string("a.out");
    p0 /= std::string("a.out");
    EXPECT_EQ(p0.string(), p0.string());
    EXPECT_EQ(p0.string(), "/home/foo/a.out");

    auto p2 = fs::path("./foo/b.txt");
    EXPECT_EQ(p2.extension(), std::string(".txt"));
    auto p3 = p2.replace_extension(".exe");
    EXPECT_EQ(p3.string(), "./foo/b.exe");

    EXPECT_EQ(fs::path("/home/foo").extension(), std::string(""));
    EXPECT_EQ(fs::path("/home/foo").replace_extension(".jpeg").string(),
              std::string("/home/foo.jpeg"));

    auto p4 = fs::path("/home/foo/a.out");
    EXPECT_EQ(p4.parent_path().string(), "/home/foo");
    EXPECT_EQ(p4.parent_path().parent_path().string(), "/home");
    EXPECT_EQ(p4.filename().string(), "a.out");
    EXPECT_EQ(p4.parent_path().filename().string(), "foo");
    EXPECT_EQ(fs::path("foo").filename().string(), "foo");

    EXPECT_TRUE(fs::exists("test_bmf_module_sdk"));
    EXPECT_FALSE(fs::is_directory("test_bmf_module_sdk"));
    EXPECT_FALSE(fs::exists("not_exists"));
}

#ifndef BMF_ENABLE_MOBILE //

TEST(module_manager, resolve_module_info) {
    auto &M = ModuleManager::instance();

    // resolve module info from builtin(ffmpeg-based)
    {
        auto info = M.resolve_module_info("c_ffmpeg_decoder");
        ASSERT_TRUE(info != nullptr);
        EXPECT_EQ(info->module_name, "c_ffmpeg_decoder");
        EXPECT_TRUE(fs::exists(info->module_path));
        EXPECT_EQ(info->module_entry, "libbuiltin_modules.CFFDecoder");
        EXPECT_EQ(info->module_type, "c++");
    }

    // resolve module info from builtin(c++)
    {
        auto info = M.resolve_module_info("pass_through");
        ASSERT_TRUE(info != nullptr);
        EXPECT_EQ(info->module_name, "pass_through");
        EXPECT_TRUE(fs::exists(info->module_path));
        EXPECT_EQ(info->module_entry, "libpass_through.PassThroughModule");
        EXPECT_EQ(info->module_type, "c++");
    }

    // resolve module info from builtin(python)
    {
        auto info = M.resolve_module_info("cpu_gpu_trans_module");
        ASSERT_TRUE(info != nullptr);
        EXPECT_EQ(info->module_name, "cpu_gpu_trans_module");
        EXPECT_TRUE(fs::exists(info->module_path));
        EXPECT_EQ(info->module_entry, "cpu_gpu_trans_module.cpu_gpu_trans_module");
        EXPECT_EQ(info->module_type, "python");
    }

    // resolve module info from sys repo(c++)
    {
        auto info = M.resolve_module_info("test_cpp_module");
        ASSERT_TRUE(info != nullptr);
        EXPECT_EQ(info->module_type, "c++");
        EXPECT_EQ(info->module_name, "test_cpp_module");
        EXPECT_TRUE(fs::exists(info->module_path));
        EXPECT_EQ(info->module_entry, "libcopy_module.CopyModule");
    }

    // resolve module info from sys repo(python)
    {
        auto info = M.resolve_module_info("test_python_module");
        ASSERT_TRUE(info != nullptr);
        EXPECT_EQ(info->module_type, "python");
        EXPECT_EQ(info->module_name, "test_python_module");
        EXPECT_TRUE(fs::exists(info->module_path));
        EXPECT_EQ(info->module_entry, "my_module.my_module");
    }

    // resolve module info from sys repo(go)
    {
        auto info = M.resolve_module_info("test_go_module");
        ASSERT_TRUE(info != nullptr);
        EXPECT_EQ(info->module_type, "go");
        EXPECT_EQ(info->module_name, "test_go_module");
        EXPECT_TRUE(fs::exists(info->module_path));
        EXPECT_EQ(info->module_entry, "test_go_module.PassThrough");
    }
}

TEST(module_manager, load_module) {
    auto &M = ModuleManager::instance();

    // load builtin module(ffmpeg-based)
    {
        auto facotry = M.load_module("c_ffmpeg_decoder");
        ASSERT_TRUE(facotry != nullptr);
        auto module = facotry->make(1);
        EXPECT_TRUE(module != nullptr);
    }

    // load builtin module(c++)
    {
        auto facotry = M.load_module("pass_through");
        ASSERT_TRUE(facotry != nullptr);
        auto module = facotry->make(1);
        EXPECT_TRUE(module != nullptr);
    }

    //load builtin module(python)
    {
        auto facotry = M.load_module("cpu_gpu_trans_module");
        ASSERT_TRUE(facotry != nullptr);
        auto module = facotry->make(1);
        EXPECT_TRUE(module != nullptr);
    }

    //load sys repo module(c++)
    {
        auto facotry = M.load_module("test_cpp_module");
        ASSERT_TRUE(facotry != nullptr);
        auto module = facotry->make(1);
        EXPECT_TRUE(module != nullptr);
    }

    //load sys repo module(python)
    {
        auto facotry = M.load_module("test_python_module");
        ASSERT_TRUE(facotry != nullptr);
        auto module = facotry->make(1);
        EXPECT_TRUE(module != nullptr);
    }

    //load sys repo module(go)
    {
        auto facotry = M.load_module("test_go_module");
        ASSERT_TRUE(facotry != nullptr);
        auto module = facotry->make(1);
        EXPECT_TRUE(module != nullptr);
    }

}

#endif // BMF_ENABLE_MOBILE

TEST(module_manager, in_app_module) {
    auto &M = ModuleManager::instance();
    auto factory = M.load_module("InAppModuleDemo", "c++");
    ASSERT_TRUE(factory != nullptr);
    auto module = factory->make();
    ASSERT_TRUE(module != nullptr);
}
