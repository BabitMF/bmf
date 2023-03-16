#include "cpp_test_helper.h"

MediaInfo::MediaInfo(std::string filepath) {
    // Execute ffprobe
    char buffer[128];
    std::string result = "";
    std::string cmd = "ffprobe -hide_banner -loglevel quiet -print_format json -show_format -show_streams ";
    FILE* pipe = popen((cmd + filepath).c_str(), "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    filePath = filepath;
    try {
        while (fgets(buffer, sizeof buffer, pipe) != NULL) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);

    std::cout << result << std::endl;

    // Deserialize results
    mediaJson = bmf_nlohmann::json::parse(result);
}

bool MediaInfo::MediaCompareEquals(std::string expected) {
    std::vector<std::string> expected_comps;
    std::stringstream mcps(expected);
    std::string token;
    while(std::getline(mcps, token, '|')) {
        expected_comps.push_back(token);
    }

    bmf_nlohmann::json extraInfo;
    if (expected_comps.size() == 9)
        extraInfo = bmf_nlohmann::json::parse(expected_comps[8]);

    for (int i = 0; i < mediaJson["streams"].size(); i++) {
        if (mediaJson["streams"][i]["codec_type"] == "video") {
            // Check video stream
            if (mediaJson["streams"][i]["height"].get<float>() != std::stof(expected_comps[1])) {
                std::cout << "Invalid height: " << mediaJson["streams"][i]["height"] << " != " << expected_comps[1] <<std::endl;
                return false;
            }

            if (mediaJson["streams"][i]["width"].get<float>() != std::stof(expected_comps[2])) {
                std::cout << "Invalid width: " << mediaJson["streams"][i]["width"] << " != " << expected_comps[2] <<std::endl;
                return false;
            }

            if (mediaJson["streams"][i]["codec_name"].get<std::string>() != expected_comps[7]) {
                std::cout << "Invalid encoding: " << mediaJson["streams"][i]["codec_name"] << " != " << expected_comps[7] <<std::endl;
                return false;
            }

            if (!extraInfo.is_null() && !extraInfo["fps"].is_null()) {
                std::string fps;
                if (!mediaJson["streams"][i]["avg_frame_rate"].is_null())
                    fps = mediaJson["streams"][i]["avg_frame_rate"];
                else
                    fps = "-1/1";

                std::vector<int> fps_comps;
                std::stringstream sfps(fps);
                for (int i; sfps >> i;) {
                    fps_comps.push_back(i);
                    if (sfps.peek() == '/')
                        sfps.ignore();
                }

                float fps_f = 0;
                if (fps_comps.size() == 1)
                    fps_f = float(fps_comps[0]);
                else if (fps_comps.size() > 2)
                    throw std::runtime_error("Invalid fraction in expected media output");
                else if (fps_comps.size() != 0) {
                    if (fps_comps[1] != 0)
                        fps_f = float(fps_comps[0]) / float(fps_comps[1]);
                }
                
                float expected_fps = std::stof(extraInfo["fps"].get<std::string>());
                float expected_diff_fps = expected_fps * 0.1;
                float actual_diff_fps = std::abs(fps_f - expected_fps);
                if (actual_diff_fps > expected_diff_fps) {
                    std::cout << "Invalid FPS: " << mediaJson["streams"][i]["avg_frame_rate"] << " != " << expected_comps[8] <<std::endl;
                    return false;
                }
            }
        }
    }

    float expected_duration = std::stof(expected_comps[3]);
    float expected_diff_duration = expected_duration * 0.1;
    float actual_diff_duration = std::abs(std::stof(mediaJson["format"]["duration"].get<std::string>()) - expected_duration);
    if (actual_diff_duration > expected_diff_duration) {
        std::cout << "Invalid duration: " << mediaJson["format"]["duration"] << " != " << expected_comps[3] <<std::endl;
        return false;
    }
    
    std::string format_name = mediaJson["format"]["format_name"].get<std::string>();
    std::string expected_format_name = expected_comps[4];
    std::transform(format_name.begin(), format_name.end(), format_name.begin(), ::toupper);
    std::transform(expected_format_name.begin(), expected_format_name.end(), expected_format_name.begin(), ::toupper);
    if (format_name != expected_format_name) {
        std::cout << "Invalid format: " << mediaJson["format"]["format_name"] << " != " << expected_comps[4] <<std::endl;
        return false;
    }

    float expected_bitrate = std::stof(expected_comps[5]);
    float expected_diff_bitrate = expected_bitrate * 0.2;
    float actual_diff_bitrate = std::abs(
        std::stof(mediaJson["format"]["bit_rate"].get<std::string>()) - expected_bitrate);
    if (actual_diff_bitrate > expected_diff_bitrate) {
        std::cout << "Invalid bitrate: " << mediaJson["format"]["bit_rate"] << " != " << expected_comps[5] <<std::endl;
        return false;
    }

    float expected_size = std::stof(expected_comps[6]);
    float expected_diff_size = expected_size * 0.2;
    float actual_diff_size = std::abs(std::stof(mediaJson["format"]["size"].get<std::string>()) - expected_size);
    if (actual_diff_size > expected_diff_size) {
        std::cout << "Invalid size: " << mediaJson["format"]["size"] << " != " << expected_comps[6] <<std::endl;
        return false;
    }

    return true;
}

 bool MediaInfo::MediaCompareMD5(const std::string& md5) {
    
    std::string md5_value;

    std::ifstream file(filePath.c_str(), std::ifstream::binary);
    if (!file)
    {
        return false;
    }
    MD5_CTX md5Context;
    MD5_Init(&md5Context);
    char buf[1024 * 16];
    while (file.good()) {
        file.read(buf, sizeof(buf));
        MD5_Update(&md5Context, buf, file.gcount());
    }

    unsigned char result[MD5_DIGEST_LENGTH];
    MD5_Final(result, &md5Context);

    char hex[35];
    memset(hex, 0, sizeof(hex));
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
    {
        sprintf(hex + i * 2, "%02x", result[i]);
    }
    hex[32] = '\0';
    md5_value = std::string(hex);
    
    if (md5_value.compare(md5) == 0) {
        return true;
    }

    return false;

 }

