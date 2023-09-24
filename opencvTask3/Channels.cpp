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
//    for (int i = 0; i < width; i++) {
//        for (int j = 0; j < height; j++) {
//            red = img.at<Vec3b>(j, i)[2];
//            img.at<Vec3b>(j, i)[2] = img.at<Vec3b>(j, i)[0];     // red -> blue
//            img.at<Vec3b>(j, i)[0] = red;
//        }
//    }
//
//    imshow("changed", img);
//    waitKey(0);
//
//    return 0;
//}
