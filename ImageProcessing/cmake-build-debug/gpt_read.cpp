//Mat doubleThrCon(Mat &src, uchar high, uchar low)
//{
//    vector<Mat> grad;
//    // 梯度通道和方向通道
//    split(src, grad);
//    for (int i = 0; i < grad[0].rows; i++)
//        for (int j = 0; j < grad[0].cols; j++)
//        {
//            if (grad[0].at<uchar>(i, j) > high)
//            {
//                grad[0].at<uchar>(i, j) = 255;
//                // 标记强边缘点的位置
//                grad[1].at<uchar>(i, j) = 2;
//            }
//            else if (grad[0].at<uchar>(i, j) > low)
//            {
//                grad[0].at<uchar>(i, j) = 0;
//                // 标记弱边缘点的位置
//                grad[1].at<uchar>(i, j) = 1;
//            }
//            else
//            {
//                grad[0].at<uchar>(i, j) = 0;
//                grad[1].at<uchar>(i, j) = 0;
//            }
//        }
//    // 真实的边缘会在弱边缘点的邻域内存在强边缘点
//    for (int i = 0; i < grad[0].rows; i++)
//        for (int j = 0; j < grad[0].cols; j++)
//        {
//            if (grad[1].at<uchar>(i, j) == 1)
//            {
//                for (int n = -1; n <= 1; n++)
//                    for (int m = -1; m <= 1; m++)
//                    {
//                        if (i + n >= 0 && j + m >= 0 && i + n < src.rows && j + m < src.cols && grad[1].at<uchar>(i + n, j + m) == 2)
//                            grad[0].at<uchar>(i, j) = 255;
//                    }
//            }
//        }
//
//    return grad[0];
//
//}
//————————————————
//版权声明：本文为CSDN博主「Thomas cs」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
//原文链接：https://blog.csdn.net/minjiuhong/article/details/89320225