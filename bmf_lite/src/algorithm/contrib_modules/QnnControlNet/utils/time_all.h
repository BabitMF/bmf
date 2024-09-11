//
// on 2023/8/14.
//

#ifndef STABLEDIFFUSION_TIME_ALL_H
#define STABLEDIFFUSION_TIME_ALL_H
#include <unordered_map>
extern std::unordered_map<int, float *> time_embedding_input_map_50;
extern std::unordered_map<int, float *> time_embedding_input_map_20;
extern std::unordered_map<int, std::unordered_map<int, float *>>
    time_embedding_input_map;

#endif // STABLEDIFFUSION_TIME_ALL_H
