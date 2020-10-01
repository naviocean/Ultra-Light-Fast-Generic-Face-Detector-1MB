#ifndef Vectoriztion_hpp
#define Vectoriztion_hpp

#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "net.h"

#define num_featuremap 4

class Vectorization
{
public:
    Vectorization(std::string &mnn_path,
                  int input_width, int input_length, int num_thread_ = 4);

    int run(unsigned char *raw_img, int img_w, int img_h, int img_c, std::vector<float> &embed);

private:
    Inference_engine ultra_net;

    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;

    float mean_vals[3] = {0.0, 0.0, 0.0};
    float norm_vals[3] = {1.0 / 255, 1.0 / 255, 1.0 / 255};
};

#endif
