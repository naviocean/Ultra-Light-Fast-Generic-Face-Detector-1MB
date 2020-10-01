#ifndef _NET_H_
#define _NET_H_

#include <vector>
#include <string>
#include <ImageProcess.hpp>
#include <Interpreter.hpp>
#include <Tensor.hpp>
#include <memory>
#include "opencv2/opencv.hpp"

class Inference_engine_tensor
{
public:
    Inference_engine_tensor()
    {
    }

    ~Inference_engine_tensor()
    {
    }

    void add_name(std::string &layer)
    {
        layer_name.push_back(layer);
    }
    std::shared_ptr<float> output(int idx)
    {
        return out_feat[idx];
    }

public:
    std::vector<std::string> layer_name;
    std::vector<std::shared_ptr<float>> out_feat;
};

class Inference_engine
{
public:
    Inference_engine();
    ~Inference_engine();

    int load_param(std::string &file, int num_thread = 1);
    int set_params(int inType, int outType, float *mean, float *scale);
    int infer_img(unsigned char *data, int img_w, int img_h, int img_c, int dstw, int dsth, Inference_engine_tensor &out);

private:
    MNN::Interpreter *netPtr;
    MNN::Session *sessionPtr;
    MNN::CV::ImageProcess::Config config;
};
#endif