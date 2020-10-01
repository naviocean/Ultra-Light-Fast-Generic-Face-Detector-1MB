//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

#include "UltraFace.hpp"
#include "Vectorization.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char **argv)
{
    // if (argc <= 3)
    // {
    //     fprintf(stderr, "Usage: %s <mnn .mnn> [image num_class files...]\n", argv[0]);
    //     return 1;
    // }

    // string mnn_path = argv[1];
    // string num_class = argv[2];
    string num_class = "2";
    string face_path = "/Users/naviocean/Working/AITOKYOLAB/Project8/Ultra-Light-Fast-Generic-Face-Detector-1MB/MNN/build/slim-320.mnn";
    string vec_path = "/Users/naviocean/Working/AITOKYOLAB/Project8/Ultra-Light-Fast-Generic-Face-Detector-1MB/MNN/build/MobileFaceNetv2_ft_JINS.mnn";

    Vectorization vectorize(vec_path, 128, 128, 4);

    UltraFace ultraface(face_path, 320, 240, 4, 0.7); // config model input

    // for (int i = 3; i < argc; i++)
    // {
    // string image_file = argv[i];
    string image_file = "/Users/naviocean/Working/AITOKYOLAB/Project8/Ultra-Light-Fast-Generic-Face-Detector-1MB/MNN/build/2.jpg";
    cout << "Processing " << image_file << endl;

    cv::Mat frame = cv::imread(image_file);
    unsigned char *frame_data = frame.data;

    auto start = chrono::steady_clock::now();
    vector<FaceInfo> face_info;
    ultraface.detect(frame_data, frame.cols, frame.rows, frame.channels(), face_info, stoi(num_class));
    // vector<float> embed;
    // vectorize.run(frame, embed);
    for (auto face : face_info)
    {
        cv::Point pt1(face.x1, face.y1);
        cv::Point pt2(face.x2, face.y2);
        cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "all time: " << elapsed.count() << " s" << endl;
    cv::imshow("UltraFace", frame);
    cv::waitKey();
    string result_name = "result.jpg";
    cv::imwrite(result_name, frame);
    // }
    return 0;
}
