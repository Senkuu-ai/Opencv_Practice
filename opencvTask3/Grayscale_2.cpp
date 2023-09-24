//#include <iostream>
//#include "opencv2/opencv.hpp"
//
//using namespace cv;
//using namespace std;
//
//int main() {
//    string path = "../lena.jpg";
//    Mat img = imread(path);
//
//    int width = img.cols;
//    int height = img.rows;
//    int red, blue;
//
//    Mat img_out = Mat::zeros(height, width, CV_8U);
//
//    for (int i = 0; i < width; i++) {
//        for (int j = 0; j < height; j++) {
//            img_out.at<uchar>(j, i) =
//                    0.2126 * img.at<Vec3b>(j, i)[0] + 0.7152 * img.at<Vec3b>(j, i)[1] + 0.0722 * img.at<Vec3b>(j, i)[2];
//        }
//    }
//
//    imshow("changed", img_out);
//    waitKey(0);
//
//    return 0;
//}
