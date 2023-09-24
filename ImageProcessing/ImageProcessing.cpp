#include <opencv2/opencv.hpp>
//#define DEBUG

using namespace cv;
using std::vector;
using std::cout;
using std::endl;

String path = "../test/";
//const double PI = acos(-1);

enum {
    BLUR,
    MINBLUR,
    MAXBLUR,
    MEDBLUR,
    GAUSSBLUR,
    SOBEL,
    LAPLACE,
    CANNY
};


namespace first_week {
    // 基本要求：使用3x3卷积核即可
    class ImageProcess {
        Mat image;
        Mat gray;
    public:
        ImageProcess() {}

        void captureImage(String path) {
            image = imread(path);
            cvtColor(image, gray, COLOR_BGR2GRAY);
        }

        // 卷积函数
        void convolution(Mat &output, Mat &kernel) {
            // 要求不使用OpenCV函数，自己写一个卷积函数，达到如以下函数的效果
            // filter2D(gray, output, gray.depth(), kernel);
            int kernelSize = kernel.rows;
            int height = output.rows;
            int width = output.cols;

            int kernelCenter = kernelSize / 2;

            // 创建一个临时矩阵用于存储卷积结果
            Mat result = output.clone();

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    Vec3b sum(0, 0, 0);

                    for (int ky = 0; ky < kernelSize; ++ky) {
                        for (int kx = 0; kx < kernelSize; ++kx) {
                            int imgX = x + kx - kernelCenter;
                            int imgY = y + ky - kernelCenter;

                            if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                                Vec3b pixel = output.at<Vec3b>(imgY, imgX);
                                double kernelValue = kernel.at<double>(ky, kx);

                                sum[0] += pixel[0] * kernelValue;
                                sum[1] += pixel[1] * kernelValue;
                                sum[2] += pixel[2] * kernelValue;
                            }
                        }
                    }
                    result.at<Vec3b>(y, x) = sum;
                }
            }
            result.copyTo(output);
        }

        // 均值滤波
        void myBlur(Mat &output, int kernelSize) {
            Mat img_copy = output.clone();
            int halfSize = kernelSize / 2;

            //遍历像素
            for (int i = 0; i < img_copy.cols; ++i) {
                for (int j = 0; j < img_copy.rows; ++j) {
                    // 卷积核
                    Vec3i sum(0, 0, 0);
                    int count = 0;
                    for (int m = -halfSize; m <= halfSize; ++m) {
                        for (int n = -halfSize; n <= halfSize; ++n) {
                            int x = i + m;
                            int y = j + n;
                            if (x >= 0 && y >= 0 && x < img_copy.cols && y < img_copy.rows) {
                                Vec3b pixel = img_copy.at<Vec3b>(x, y);
                                sum += pixel;
                                count++;
                            }
                        }
                    }
                    // 保存该像素计算
                    output.at<Vec3b>(i, j) = sum / count;
                }
            }
        }

        // 最小值滤波
        void myMinBlur(Mat &output, int kernelSize) {
            Mat img_copy = output.clone();
            int halfSize = kernelSize / 2;

            //遍历像素
            for (int i = 0; i < img_copy.cols; ++i) {
                for (int j = 0; j < img_copy.rows; ++j) {
                    // 卷积核
                    Vec3i min(255, 255, 255);
                    for (int m = -halfSize; m <= halfSize; ++m) {
                        for (int n = -halfSize; n <= halfSize; ++n) {
                            int x = i + m;
                            int y = j + n;
                            if (x >= 0 && y >= 0 && x < img_copy.cols && y < img_copy.rows) {
                                for (int k = 0; k < 3; ++k) {
                                    if (img_copy.at<Vec3b>(x, y)[k] < min[k]) {
                                        min[k] = img_copy.at<Vec3b>(x, y)[k];
                                    }
                                }
                            }
                        }
                    }
                    // 保存该像素计算
                    output.at<Vec3b>(i, j) = min;
                }
            }
        }

        // 最大值滤波
        void myMaxBlur(Mat &output, int kernelSize) {
            Mat img_copy = output.clone();
            int halfSize = kernelSize / 2;

            //遍历像素
            for (int i = 0; i < img_copy.cols; ++i) {
                for (int j = 0; j < img_copy.rows; ++j) {
                    // 卷积核
                    Vec3i max(0, 0, 0);
                    for (int m = -halfSize; m <= halfSize; ++m) {
                        for (int n = -halfSize; n <= halfSize; ++n) {
                            int x = i + m;
                            int y = j + n;
                            if (x >= 0 && y >= 0 && x < img_copy.cols && y < img_copy.rows) {
                                for (int k = 0; k < 3; ++k) {
                                    if (img_copy.at<Vec3b>(x, y)[k] > max[k]) {
                                        max[k] = img_copy.at<Vec3b>(x, y)[k];
                                    }
                                }
                            }
                        }
                    }
                    // 保存该像素计算
                    output.at<Vec3b>(i, j) = max;
                }
            }
        }

        // 计算中值
        int calculateMedian(int *data, int dataSize) {
            std::sort(data, data + dataSize);

            // 计算中值
            int median;
            if (dataSize % 2 == 0) {
                // 偶数取平均值
                median = (data[dataSize / 2 - 1] + data[dataSize / 2]) / 2;
            } else {
                // 奇数
                median = data[dataSize / 2];
            }
            return median;
        }

        // 中值滤波
        void myMediumBlur(Mat &output, int kernel_size) {
            Mat img_copy = output.clone();
            int height = output.rows, width = output.cols;
            for (int i = 0; i < height - kernel_size; i++) {
                for (int j = 0; j < width - kernel_size; j++) {
                    int data[3][kernel_size * kernel_size]; // 存储像素数据的数组
                    int k_seq = 0; // 用于迭代data数组的索引
                    for (int n = 0; n < kernel_size; n++) {
                        for (int m = 0; m < kernel_size; m++) {
                            for (int k = 0; k < 3; k++) {
                                data[k][k_seq] = img_copy.at<Vec3b>(i + n, j + m)[k];
                            }
                            k_seq++;
                        }
                    }
                    // 计算中值并赋值给输出图像的像素
                    output.at<Vec3b>(i + kernel_size / 2, j + kernel_size / 2)[0] = calculateMedian(data[0],
                                                                                                    kernel_size *
                                                                                                    kernel_size);
                    output.at<Vec3b>(i + kernel_size / 2, j + kernel_size / 2)[1] = calculateMedian(data[1],
                                                                                                    kernel_size *
                                                                                                    kernel_size);
                    output.at<Vec3b>(i + kernel_size / 2, j + kernel_size / 2)[2] = calculateMedian(data[2],
                                                                                                    kernel_size *
                                                                                                    kernel_size);
                }
            }
        }

        // 生成高斯核
        // 生成高斯核
        Mat gaussKernel(int size, double sigma) {
            Mat kernel(size, size, CV_64F);
            double twoSigmaSquared = 2.0 * sigma * sigma;
            double kernelSum = 0.0;

            int kernelCenter = size / 2;

            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    int x = i - kernelCenter;
                    int y = j - kernelCenter;
                    double exponent = -(x * x + y * y) / twoSigmaSquared;
                    kernel.at<double>(i, j) = exp(exponent) / (M_PI * twoSigmaSquared);
                    kernelSum += kernel.at<double>(i, j);
                }
            }

            // 正规化核，使总和等于1
            kernel /= kernelSum;

            return kernel;
        }

        // 高斯滤波
        void myGaussBlur(Mat &output) {
            // 基本要求：不用自己生成高斯核，查一下网上常用的高斯核卷积即可
            // 如果你想折腾自己生成高斯核也可，函数任你修改
            int kernelSize = 5; // 核的大小（奇数）
            double sigma = 1.0; // 高斯核的标准差
            Mat Gauss_kernel = gaussKernel(kernelSize, sigma);

            convolution(output, Gauss_kernel);
        }

        // sobel 边缘检测
        void mySobel(Mat &output) {
            Mat X_kernel = (Mat_<float>(3, 3) <<
                                              -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1);
            Mat Y_kernel = (Mat_<float>(3, 3) <<
                                              1, 2, 1,
                    0, 0, 0,
                    -1, -2, -1);
            Mat img_gray;
            cvtColor(output, img_gray, COLOR_BGR2GRAY);
            cvtColor(output, output, COLOR_BGR2GRAY);
            int kernelSize = 3;
            int halfSize = kernelSize / 2;

            //遍历像素
            for (int i = 0; i < img_gray.cols; ++i) {
                for (int j = 0; j < img_gray.rows; ++j) {
                    // 卷积核
                    Vec3i max(0, 0, 0);
                    int Gx = 0, Gy = 0;
                    for (int m = -halfSize; m <= halfSize; ++m) {
                        for (int n = -halfSize; n <= halfSize; ++n) {
                            int x = i + m;
                            int y = j + n;
                            if (x >= 0 && y >= 0 && x < img_gray.cols && y < img_gray.rows) {
                                Gx += img_gray.at<uchar>(x, y) * X_kernel.at<float>(m + 1, n + 1);
                                Gy += img_gray.at<uchar>(x, y) * Y_kernel.at<float>(m + 1, n + 1);

                            }
                        }
                    }
                    // 保存该像素计算
                    output.at<uchar>(i, j) = saturate_cast<uchar>(0.5 * (abs(Gx) + abs(Gy)));

                }
            }
        }

        // laplace算子
        void myLaplace(Mat &output) {
            Mat Laplace_kernel = (Mat_<float>(3, 3) <<
                                                    0.0, 1.0, 0.0,
                    1.0, -4.0, 1.0,
                    0.0, 1.0, 0.0);
            Mat img_gray;

            cvtColor(output, img_gray, COLOR_BGR2GRAY);
            cvtColor(output, output, COLOR_BGR2GRAY);
            int kernelSize = 3;
            int halfSize = kernelSize / 2;

            //遍历像素
            for (int i = 0; i < img_gray.cols; ++i) {
                for (int j = 0; j < img_gray.rows; ++j) {
                    // 卷积核
                    Vec3i max(0, 0, 0);
                    int sum = 0;
                    for (int m = -halfSize; m <= halfSize; ++m) {
                        for (int n = -halfSize; n <= halfSize; ++n) {
                            int x = i + m;
                            int y = j + n;
                            if (x >= 0 && y >= 0 && x < img_gray.cols && y < img_gray.rows) {
                                sum += img_gray.at<uchar>(x, y) * Laplace_kernel.at<float>(m + 1, n + 1);
                            }
                        }
                    }
                    // 保存该像素计算
                    output.at<uchar>(i, j) = saturate_cast<uchar>(sum);

                }
            }
        }

        // 带有方向信息的sobel，为canny准备
        Mat sobelFilter(Mat &src) {
            Mat value = src.clone();
            Mat direct = src.clone();
            Mat X_kernel = (Mat_<float>(3, 3) <<
                                              -1, 0, 1,
                    -2, 0, 2,
                    -1, 0, 1);
            Mat Y_kernel = (Mat_<float>(3, 3) <<
                                              1, 2, 1,
                    0, 0, 0,
                    -1, -2, -1);
            int kernelSize = 3;
            int halfSize = kernelSize / 2;

            //遍历像素
            for (int i = 0; i < src.cols; ++i) {
                for (int j = 0; j < src.rows; ++j) {
                    // 卷积核
                    Vec3i max(0, 0, 0);
                    int Gx = 0, Gy = 0;
                    for (int m = -halfSize; m <= halfSize; ++m) {
                        for (int n = -halfSize; n <= halfSize; ++n) {
                            int x = i + m;
                            int y = j + n;
                            if (x >= 0 && y >= 0 && x < src.cols && y < src.rows) {
                                Gx += src.at<uchar>(x, y) * X_kernel.at<float>(m + 1, n + 1);
                                Gy += src.at<uchar>(x, y) * Y_kernel.at<float>(m + 1, n + 1);

                            }
                        }
                    }
                    // 保存该像素计算
                    value.at<uchar>(i, j) = saturate_cast<uchar>(sqrt(Gx * Gx + Gy * Gy));
                    direct.at<uchar>(i, j) = static_cast<uchar>(atan(Gy / (Gx + 0.001)) * 8 / CV_PI + 4);
                }
            }
            imshow("sobel", value);
            Mat temp[] = {value, direct};
            Mat result;
            merge(temp, 2, result);

            return result;
        }

        void NMS(Mat &src) {
            vector<Mat> grad;
            split(src, grad);
            Mat gradValue = grad[0].clone();

            for (int i = 0; i < gradValue.rows; i++)
                for (int j = 0; j < gradValue.cols; j++) {
                    if (grad[1].at<uchar>(i, j) == 0 || grad[1].at<uchar>(i, j) == 7) {
                        if (i < gradValue.rows - 1)
                            if (gradValue.at<uchar>(i, j) <= gradValue.at<uchar>(i + 1, j)) {
                                grad[0].at<uchar>(i, j) = 0;
                                continue;
                            }
                        if (i > 0)
                            if (gradValue.at<uchar>(i, j) <= gradValue.at<uchar>(i - 1, j)) {
                                grad[0].at<uchar>(i, j) = 0;
                                continue;
                            }
                    }
                    if (grad[1].at<uchar>(i, j) == 1 || grad[1].at<uchar>(i, j) == 2) {
                        if (i < gradValue.rows - 1 && j < gradValue.cols - 1)
                            if (gradValue.at<uchar>(i, j) <= gradValue.at<uchar>(i + 1, j + 1)) {
                                grad[0].at<uchar>(i, j) = 0;
                                continue;
                            }
                        if (i > 0 && j > 0)
                            if (gradValue.at<uchar>(i, j) <= gradValue.at<uchar>(i - 1, j - 1)) {
                                grad[0].at<uchar>(i, j) = 0;
                                continue;
                            }
                    }
                    if (grad[1].at<uchar>(i, j) == 3 || grad[1].at<uchar>(i, j) == 4) {
                        if (j < gradValue.cols - 1)
                            if (gradValue.at<uchar>(i, j) <= gradValue.at<uchar>(i, j + 1)) {
                                grad[0].at<uchar>(i, j) = 0;
                                continue;
                            }
                        if (j > 0)
                            if (gradValue.at<uchar>(i, j) <= gradValue.at<uchar>(i, j - 1)) {
                                grad[0].at<uchar>(i, j) = 0;
                                continue;
                            }
                    }
                    if (grad[1].at<uchar>(i, j) == 5 || grad[1].at<uchar>(i, j) == 6) {
                        if (i < gradValue.rows - 1 && j > 0)
                            if (gradValue.at<uchar>(i, j) <= gradValue.at<uchar>(i + 1, j - 1)) {
                                grad[0].at<uchar>(i, j) = 0;
                                continue;
                            }
                        if (i > 0 && j < gradValue.cols - 1)
                            if (gradValue.at<uchar>(i, j) <= gradValue.at<uchar>(i - 1, j + 1)) {
                                grad[0].at<uchar>(i, j) = 0;
                                continue;
                            }
                    }
                }
            imshow("NMS", grad[0]);
            merge(grad, src);
        }


        //双阈值连接
        Mat doubleThrCon(Mat &src, uchar high, uchar low) {
            vector<Mat> grad;
            // 梯度通道和方向通道
            split(src, grad);
            for (int i = 0; i < grad[0].rows; i++)
                for (int j = 0; j < grad[0].cols; j++) {
                    if (grad[0].at<uchar>(i, j) > high) {
                        grad[0].at<uchar>(i, j) = 255;
                        // 标记强边缘点的位置
                        grad[1].at<uchar>(i, j) = 2;
                    } else if (grad[0].at<uchar>(i, j) > low) {
                        grad[0].at<uchar>(i, j) = 0;
                        // 标记弱边缘点的位置
                        grad[1].at<uchar>(i, j) = 1;
                    } else {
                        grad[0].at<uchar>(i, j) = 0;
                        grad[1].at<uchar>(i, j) = 0;
                    }
                }
            // 真实的边缘会在弱边缘点的邻域内存在强边缘点
            for (int i = 0; i < grad[0].rows; i++)
                for (int j = 0; j < grad[0].cols; j++) {
                    if (grad[1].at<uchar>(i, j) == 1) {
                        for (int n = -1; n <= 1; n++)
                            for (int m = -1; m <= 1; m++) {
                                if (i + n >= 0 && j + m >= 0 && i + n < src.rows && j + m < src.cols &&
                                    grad[1].at<uchar>(i + n, j + m) == 2)
                                    grad[0].at<uchar>(i, j) = 255;
                            }
                    }
                }
            return grad[0];
        }

        // Canny算子，查询canny的四个步骤，试试不调用opencv能否自己实现
        void myCanny(Mat &output) {
            myGaussBlur(output);
            Mat grad;
            cvtColor(output, output, COLOR_BGR2GRAY);
            grad = sobelFilter(output);
            NMS(grad);
            uchar high = 70,low=60;
            output = doubleThrCon(grad,high,low);
        }
    };
}

int main() {
    first_week::ImageProcess image_process;
    //image_process.captureImage(path + "00A.png");
    Mat output = imread("../lena.jpg");
    Mat origin = output.clone();


    int op = CANNY;
    switch (op) {
        default:
            break;
        case BLUR:
            image_process.myBlur(output, 3);
            break;
        case MINBLUR:
            image_process.myMinBlur(output, 3);
            break;
        case MAXBLUR:
            image_process.myMaxBlur(output, 3);
            break;
        case MEDBLUR:
            image_process.myMediumBlur(output, 3);
            break;
        case GAUSSBLUR:
            image_process.myGaussBlur(output);
            break;
        case SOBEL:
            image_process.mySobel(output);
            break;
        case LAPLACE:
            image_process.myLaplace(output);
            break;
        case CANNY:
            image_process.myCanny(output);
            break;
    }
    imshow("output", output);
    imshow("origin", origin);
    waitKey(0);
    return 0;
}