//#include <iostream>
//#include <opencv2/opencv.hpp> // 使用OpenCV库进行图像处理
//
//using namespace std;
//using namespace cv;
//
//Mat gammaCorrect(Mat src,float gamma){
//    Mat output = src.clone();
//    for (int i = 0; i < src.rows; i++) {
//        for (int j = 0; j < src.cols; j++) {
//            for (int k = 0; k < src.channels(); ++k) {
//                double pixel = (double)src.at<Vec3b>(i,j)[k];
//                output.at<Vec3b>(i,j)[k] = pow(pixel/255.0,gamma)*255.0;
//            }
//        }
//    }
//    return output;
//}
//
//int main() {
//    // 读取图像
//    Mat image = imread("../squirrel.jpg", IMREAD_COLOR);
//
//    // 设置Gamma值
//    double gamma = 1.8;
//
//    Mat gammaCorrected = gammaCorrect(image,gamma);
//
//    imshow("origin", image);
//    imshow("Gamma_corect", gammaCorrected);
//
//    waitKey(0);
//    return 0;
//}
