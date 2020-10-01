#include "net.h"

Inference_engine::Inference_engine()
{
}

Inference_engine::~Inference_engine()
{
    if (netPtr != NULL)
    {
        if (sessionPtr != NULL)
        {
            netPtr->releaseSession(sessionPtr);
            sessionPtr = NULL;
        }

        delete netPtr;
        netPtr = NULL;
    }
}

int Inference_engine::load_param(std::string &file, int num_thread)
{
    if (!file.empty())
    {
        if (file.find(".mnn") != std::string::npos)
        {
            netPtr = MNN::Interpreter::createFromFile(file.c_str());
            if (nullptr == netPtr)
                return -1;

            MNN::ScheduleConfig sch_config;
            sch_config.type = (MNNForwardType)MNN_FORWARD_CPU;
            if (num_thread > 0)
                sch_config.numThread = num_thread;
            sessionPtr = netPtr->createSession(sch_config);
            if (nullptr == sessionPtr)
                return -1;
        }
        else
        {
            return -1;
        }
    }

    return 0;
}

int Inference_engine::set_params(int srcType, int dstType,
                                 float *mean, float *scale)
{
    config.destFormat = (MNN::CV::ImageFormat)dstType;
    config.sourceFormat = (MNN::CV::ImageFormat)srcType;

    // mean¡¢normal
    ::memcpy(config.mean, mean, 3 * sizeof(float));
    ::memcpy(config.normal, scale, 3 * sizeof(float));

    // filterType¡¢wrap
    config.filterType = (MNN::CV::Filter)(1);
    config.wrap = (MNN::CV::Wrap)(2);

    return 0;
}

// infer
int Inference_engine::infer_img(unsigned char *data, int img_w, int img_h, int img_c, int dstw, int dsth, Inference_engine_tensor &out)
{
    MNN::Tensor *tensorPtr = netPtr->getSessionInput(sessionPtr, nullptr);
    MNN::CV::Matrix transform;
    // auto resize for full conv network.
    bool auto_resize = false;
    if (!auto_resize)
    {
        std::vector<int> dims = {1, img_c, dsth, dstw};
        netPtr->resizeTensor(tensorPtr, dims);
        netPtr->resizeSession(sessionPtr);
    }
    transform.postScale(1.0f / dstw, 1.0f / dsth);
    transform.postScale(img_w, img_h);
    std::unique_ptr<MNN::CV::ImageProcess> process(MNN::CV::ImageProcess::create(config.sourceFormat, config.destFormat, config.mean, 3, config.normal, 3));
    process->setMatrix(transform);
    process->convert(data, img_w, img_h, img_w * img_c, tensorPtr);
    netPtr->runSession(sessionPtr);

    for (int i = 0; i < out.layer_name.size(); i++)
    {
        const char *layer_name = NULL;
        if (strcmp(out.layer_name[i].c_str(), "") != 0)
        {
            layer_name = out.layer_name[i].c_str();
        }
        MNN::Tensor *tensorOutPtr = netPtr->getSessionOutput(sessionPtr, layer_name);

        std::vector<int> shape = tensorOutPtr->shape();

        auto tensor = reinterpret_cast<MNN::Tensor *>(tensorOutPtr);

        std::unique_ptr<MNN::Tensor> hostTensor(new MNN::Tensor(tensor, tensor->getDimensionType(), true));
        tensor->copyToHostTensor(hostTensor.get());
        tensor = hostTensor.get();

        auto size = tensorOutPtr->elementSize();
        std::shared_ptr<float> destPtr(new float[size * sizeof(float)]);

        ::memcpy(destPtr.get(), tensorOutPtr->host<float>(), size * sizeof(float));

        out.out_feat.push_back(destPtr);
    }

    return 0;
}