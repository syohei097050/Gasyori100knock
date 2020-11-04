#include <opencv2/opencv.hpp>

cv::Mat convBGR2RGB(cv::Mat src);
cv::Mat convBGR2Gray(cv::Mat src);
cv::Mat convBGR2HSV(cv::Mat src);
cv::Mat convHSV2BGR(cv::Mat src);
cv::Mat binarize(cv::Mat src, int th = 128);
cv::Mat binarizeOtsu(cv::Mat src);
cv::Mat invertHue(cv::Mat src);
cv::Mat decreaseColor(cv::Mat src);
cv::Mat meanPooling(cv::Mat src, int size);
cv::Mat maxPooling(cv::Mat src, int size);
cv::Mat gaussianBlur(cv::Mat src);
cv::Mat medianBlur(cv::Mat src);