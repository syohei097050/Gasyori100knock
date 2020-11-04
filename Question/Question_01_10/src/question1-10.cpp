#include <opencv2/opencv.hpp>
#include <iostream>

#include "imgproc.hpp"


int main()
{
    cv::Mat src = cv::imread("../imori.jpg");
    
    cv::Mat dst1 = convBGR2RGB(src);
    if (dst1.empty())
        printf("Error : dst1 is empty\n");
    else
        cv::imwrite("../dst1.jpg", dst1);
    
    cv::Mat dst2 = convBGR2Gray(src);
    if (dst2.empty())
        printf("Error : dst2 is empty\n");
    else
        cv::imwrite("../dst2.jpg", dst2);

    cv::Mat dst3 = binarize(src);
    if (dst3.empty())
        printf("Error : dst3 is empty\n");
    else
        cv::imwrite("../dst3.jpg", dst3);

    cv::Mat dst4 = binarizeOtsu(src);
    if (dst4.empty())
        printf("Error : dst4 is empty\n");
    else
        cv::imwrite("../dst4.jpg", dst4);

    cv::Mat dst5 = invertHue(src);
    if (dst5.empty())
        printf("Error : dst5 is empty\n");
    else
        cv::imwrite("../dst5.jpg", dst5);

    cv::Mat dst6 = decreaseColor(src);
    if (dst6.empty())
        printf("Error : dst6 is empty\n");
    else
        cv::imwrite("../dst6.jpg", dst6);

    cv::Mat dst7 = meanPooling(src, 8);
    if (dst7.empty())
        printf("Error : dst7 is empty\n");
    else
        cv::imwrite("../dst7.jpg", dst7);

    cv::Mat dst8 = maxPooling(src, 8);
    if (dst8.empty())
        printf("Error : dst8 is empty\n");
    else
        cv::imwrite("../dst8.jpg", dst8);

    cv::Mat src2 = cv::imread("../imori_noise.jpg");

    cv::Mat dst9 = gaussianBlur(src2);
    if (dst9.empty())
        printf("Error : dst9 is empty\n");
    else
        cv::imwrite("../dst9.jpg", dst9);

    cv::Mat dst10 = medianBlur(src2);
    if (dst10.empty())
        printf("Error : dst10 is empty\n");
    else
        cv::imwrite("../dst10.jpg", dst10);

    return 0;
}