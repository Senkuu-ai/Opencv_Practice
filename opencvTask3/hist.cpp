#include "opencv2/opencv.hpp"

using namespace cv;

//@para gray:需要统计的图   Hist：用于存放统计数据
void GetHist(Mat gray,Mat &Hist)    //统计8Bit量化图像的灰度直方图
{
    const int channels[1] = { 0 }; //通道索引
    float inRanges[2] = { 0,255 };  //像素范围
    const float* ranges[1] = {inRanges};//像素灰度级范围
    const int bins[1] = { 256 };   //直方图的维度
    calcHist(&gray, 1, channels,Mat(), Hist,1, bins, ranges);
}
void ShowHist(Mat &Hist)
{
    //准备绘制直方图
    int hist_w = 512;
    int hist_h = 400;
    int width = 2;

    double hist_max, hist_min;
    Point min_loc, max_loc;
    minMaxLoc(Hist,&hist_min, &hist_max, &min_loc, &max_loc);

    Mat histImage = Mat::zeros(hist_h,hist_w,CV_8UC3);   //全黑，512*400
    for (int i = 0; i < Hist.rows; i++)
    {
        rectangle(histImage,Point(width*i,hist_h-1),Point(width*(i+1),hist_h-cvRound((Hist.at<float>(i)/hist_max)*hist_h-20)),
                  Scalar(255,255,255),-1);
    }
    namedWindow("histImage", WINDOW_AUTOSIZE);
    imshow("histImage", histImage);
    //waitKey(0);
    
}
int main()
{
    Mat src,gray,hist;   //hist用于统计gray的直方图
    src=imread("../lena.jpg");
    cvtColor(src,gray,COLOR_BGR2GRAY);
    GetHist(gray,hist);
    ShowHist(hist);
    namedWindow("gray");
    imshow("gray",gray);
    waitKey(0);
    return 0;
}