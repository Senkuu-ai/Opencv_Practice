#include <opencv2/opencv.hpp>
//#define DEBUG

using namespace cv;
using std::vector;
using std::cout;
using std::endl;

String path = "../test/";
//const double PI = acos(-1);

enum
{
	BLUR,
	MINBLUR,
	MAXBLUR,
	MEDBLUR,
	GAUSSBLUR,
	SOBEL,
	LAPLACE,
	CANNY
};


namespace first_week
{
	// 基本要求：使用3x3卷积核即可
	class ImageProcess
	{
		Mat image;
		Mat gray;
	public:
		ImageProcess() {}
		void captureImage(String path) 
		{
			image = imread(path);
			cvtColor(image, gray, COLOR_BGR2GRAY);
		}
		// 卷积函数
		void convolution(Mat& output, Mat& kernel)
		{
			// 要求不使用OpenCV函数，自己写一个卷积函数，达到如以下函数的效果
			// filter2D(gray, output, gray.depth(), kernel);
			int sizek = kernel.rows, height = output.rows, width = output.cols;
			for (int i = 0; i < rows - sizek;i++)
			{
				for (int j = 0; j　 < cols - sizek; j++)
				{
					int sum[3] = { 0 };
					for (int n = 0; n < sizek; n++)
					{
						for (int m = 0; m < sizek; m++)
						{
							sum[0] += output.at<Vec3b>(i + n, j + m)[0];
							sum[1] += output.at<Vec3b>(i + n, j + m)[1];
							sum[2] += output.at<Vec3b>(i + n, j + m)[2];
						}
					}
					output.at<Vec3b>(i + sizek / 2, j + sizek / 2)[0] = sum[0] /sizek/sizek;
					output.at<Vec3b>(i + sizek / 2, j + sizek / 2)[1] = sum[1] / sizek / sizek;
					output.at<Vec3b>(i + sizek / 2, j + sizek / 2)[2] = sum[2] / sizek / sizek;
				}
			}
			imshow("test convolution", output);
			waitKey(0);
			destorywindow();
		}
		// 均值滤波
		void myBlur(Mat& output)
		{
			Mat
			blur(gray, output, Size(3, 3));
		}
		// 最小值滤波
		void myMinBlur(Mat& output)
		{

		}
		// 最大值滤波
		void myMaxBlur(Mat& output)
		{

		}
		// 中值滤波
		void myMediumBlur(Mat& output)
		{

		}
		// 高斯滤波
		void myGaussBlur(Mat& output)
		{
			// 基本要求：不用自己生成高斯核，查一下网上常用的高斯核卷积即可
			// 如果你想折腾自己生成高斯核也可，函数任你修改
		}
		// sobel 边缘检测
		void mySobel(Mat& output)
		{

		}
		// laplace算子
		void myLaplace(Mat& output)
		{
			
		}
		// Canny算子，查询canny的四个步骤，试试不调用opencv能否自己实现
		void myCanny(Mat& output)
		{

		}
	};
}

int main()
{
	first_week::ImageProcess image_process;
	image_process.captureImage(path + "00A.png");
	
	Mat output;

	int op = BLUR;
	switch (op)
	{
	default:
		break;
	case BLUR: image_process.myBlur(output); break;
	case MINBLUR:image_process.myMinBlur(output); break;
	case MAXBLUR:image_process.myMaxBlur(output); break;
	case MEDBLUR:image_process.myMediumBlur(output); break;
	case GAUSSBLUR:image_process.myGaussBlur(output); break;
	case SOBEL:image_process.mySobel(output); break;
	case LAPLACE:image_process.myLaplace(output); break;
	case CANNY:image_process.myCanny(output); break;
	}
	imshow("output", output);
	waitKey(0);
	return 0;
}