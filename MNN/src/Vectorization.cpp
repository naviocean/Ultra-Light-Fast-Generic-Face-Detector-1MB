#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

#include "Vectorization.hpp"

using namespace std;

Vectorization::Vectorization(std::string &mnn_path,
                             int input_width, int input_length, int num_thread_)
{
    num_thread = num_thread_;
    in_w = input_width;
    in_h = input_length;
    ultra_net.load_param(mnn_path, num_thread);

    ultra_net.set_params(2, 1, mean_vals, norm_vals);
}

int Vectorization::run(unsigned char *raw_img, int img_w, int img_h, int img_c, std::vector<float> &embed)
{
    image_h = img_h;
    image_w = img_w;

    auto start = chrono::steady_clock::now();

    // get output data
    Inference_engine_tensor out;
    string output = "output";
    out.add_name(output);
    ultra_net.infer_img(raw_img, img_w, img_h, img_c, in_w, in_h, out);
    float *score = out.output(0).get();
    for (int i = 0; i < 256; i++)
    {
        // std::cout << score[i] << std::endl;
        embed.push_back(score[i]);
    }
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    // cout << "inference time:" << elapsed.count() << " s" << endl;
    return 0;
}