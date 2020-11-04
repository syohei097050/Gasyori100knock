#include <vector>
#include <algorithm>

#include "imgproc.hpp"

cv::Mat convBGR2RGB(cv::Mat src)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    for (int y = 0; y < src.rows; y++){
        cv::Vec3b *src_ptr = src.ptr<cv::Vec3b>(y);
        cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>(y);
        for(int x = 0; x < src.cols; x++){
            dst_ptr[x] = cv::Vec3b(src_ptr[x][2], src_ptr[x][1], src_ptr[x][0]);
        }
    } 
    return dst;
}

cv::Mat convBGR2Gray(cv::Mat src)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    for (int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
            dst.at<uchar>(y, x) = static_cast<uchar>(0.2126 * src.at<cv::Vec3b>(y, x)[2] + 
                                                     0.7152 * src.at<cv::Vec3b>(y, x)[1] + 
                                                     0.0722 * src.at<cv::Vec3b>(y, x)[0]);
        }
    }
    return dst;
}

cv::Mat convBGR2HSV(cv::Mat src)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC3);
    for (int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){

            float b = src.at<cv::Vec3b>(y, x)[0] / 255.;
            float g = src.at<cv::Vec3b>(y, x)[1] / 255.;
            float r = src.at<cv::Vec3b>(y, x)[2] / 255.;
            float max = std::max(b, std::max(g, r));
            float min = std::min(b, std::min(g, r));

            float hue, sat, val;

            if (min == max)
                hue = 0;
            else if(min == b)
                hue = 60 * (g - r) / (max - min) + 60;
            else if(min == r)
                hue = 60 * (b - g) / (max - min) + 180;
            else if(min == g)
                hue = 60 * (r - b) / (max - min) + 300;

            val = max;
            sat = max - min;
            dst.at<cv::Vec3f>(y, x)[0] = hue;
            dst.at<cv::Vec3f>(y, x)[1] = sat;
            dst.at<cv::Vec3f>(y, x)[2] = val;
        }
    }
    return dst;
}

cv::Mat convHSV2BGR(cv::Mat src)
{
    int width = src.cols;
    int height = src.rows;

    float b, g, r;

    cv::Mat dst = cv::Mat::zeros(height, width, CV_8UC3);
    for (int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
            float hue = src.at<cv::Vec3f>(y, x)[0];
            float sat = src.at<cv::Vec3f>(y, x)[1];
            float val = src.at<cv::Vec3f>(y, x)[2];
            float C = sat;
            float hue_dash = hue / 60;
            float X = C * (1 - abs(fmod(hue_dash, 2) - 1));

            b = g = r = val - C;

            if(hue_dash < 1){
                g += X;
                r += C;
            }else if(hue_dash < 2){
                g += C;
                r += X;
            }else if(hue_dash < 3){
                b += X;
                g += C;
            }else if(hue_dash < 4){
                b += C;
                g += X;
            }else if(hue_dash < 5){
                r += X;
                b += C;
            }else if(hue_dash < 6){
                r += C;
                b += X;
            }
            dst.at<cv::Vec3b>(y, x)[0] = (uchar)(b * 255);
            dst.at<cv::Vec3b>(y, x)[1] = (uchar)(g * 255);
            dst.at<cv::Vec3b>(y, x)[2] = (uchar)(r * 255);
        }
    }
    return dst;
}

cv::Mat binarize(cv::Mat src, int th)
{
    cv::Mat gray;
    if (src.type() != CV_8UC1){
        gray = convBGR2Gray(src);
    }else{
        gray = src;
    }
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < gray.rows; y++){
        for(int x = 0; x < gray.cols; x++){
            if (gray.at<uchar>(y, x) < th)
                dst.at<uchar>(y, x) = 0;
            else
                dst.at<uchar>(y, x) = 255;
        }
    }
    return dst;
}

double calcSigma_b(cv::Mat &src, int th)
{
    int w0 = 0;             //クラス0に属する画素数
    int w1 = 0;             //クラス1に属する画素数
    int p0 = 0;             //クラス0に属する画素値の合計値
    int p1 = 0;             //クラス1に属する画素値の合計値
    double mean0, mean1;    //クラス0, 1に属する画素値の平均値
    double sigma_b;         //クラス間分散

    for (int y = 0; y < src.rows; y++){
        for (int x = 0; x < src.cols; x++){
            if (src.at<uchar>(y, x) < th){
                w0++;
                p0 += src.at<uchar>(y, x);
            }else{
                w1++;
                p1 += src.at<uchar>(y, x);
            }
        }
    }

    mean0 = double(p0) / w0;
    mean1 = double(p1) / w1;
    return w0 * w1 * pow(mean0 - mean1, 2) / (w0 + w1);
}

cv::Mat binarizeOtsu(cv::Mat src)
{
    cv::Mat gray;
    if (src.type() != CV_8UC1){
        gray = convBGR2Gray(src);
    }else{
        gray = src;
    }
    
    int th;
    int left = 0;
    int right = 255;
    double max_sifgma_b = 0;
    for(int i = 0; i < 256; i++){
        double sigma_b = calcSigma_b(src, i);
        if(sigma_b > max_sifgma_b)
            th = i;
            max_sifgma_b = sigma_b;
        //printf("th = %d, sigma_b = %f\n", i, sigma_b);
    }

    printf("Threshold : %d\n", th);
    return binarize(gray, th);
    
}

cv::Mat invertHue(cv::Mat src)
{
    cv::Mat hsv = convBGR2HSV(src);
    for (int y = 0; y < hsv.rows; y++){
        for (int x = 0; x < hsv.cols; x++){
            hsv.at<cv::Vec3f>(y,x)[0] = fmod((hsv.at<cv::Vec3f>(y,x)[0] + 180), 360);
        }
    }
    cv::Mat dst = convHSV2BGR(hsv);

    return dst;
}

uchar quantize(uchar val)
{
    val >>= 6;
    if(val == 0)
        return 32;
    else if (val == 1)
        return 96;
    else if (val == 2)
        return 160;
    else if (val == 3)
        return 224;

    return 0;
}

cv::Mat decreaseColor(cv::Mat src)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);

    for (int y = 0; y < src.rows; y++){
        for (int x = 0; x < src.cols; x++){
            dst.at<cv::Vec3b>(y, x)[0] = quantize(src.at<cv::Vec3b>(y, x)[0]);
            dst.at<cv::Vec3b>(y, x)[1] = quantize(src.at<cv::Vec3b>(y, x)[1]);
            dst.at<cv::Vec3b>(y, x)[2] = quantize(src.at<cv::Vec3b>(y, x)[2]);
        }
    }

    return dst;
}

cv::Mat meanPooling(cv::Mat src, int size)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; y+=size){
        for (int x = 0; x < src.cols; x+=size){
            int b = 0;
            int g = 0;
            int r = 0;
            for (int ky = 0; ky < size; ky++){
                for (int kx = 0; kx < size; kx++){
                    b += src.at<cv::Vec3b>(y+ky, x+kx)[0];
                    g += src.at<cv::Vec3b>(y+ky, x+kx)[1];
                    r += src.at<cv::Vec3b>(y+ky, x+kx)[2];
                }
            }
            b /= size * size;
            g /= size * size;
            r /= size * size;
            for (int ky = 0; ky < size; ky++){
                for (int kx = 0; kx < size; kx++){
                    dst.at<cv::Vec3b>(y+ky, x+kx)[0] = (uchar)b;
                    dst.at<cv::Vec3b>(y+ky, x+kx)[1] = (uchar)g;
                    dst.at<cv::Vec3b>(y+ky, x+kx)[2] = (uchar)r;
                }
            }
        }
    }

    return dst;
}

cv::Mat maxPooling(cv::Mat src, int size)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; y+=size){
        for (int x = 0; x < src.cols; x+=size){
            uchar b_max = 0;
            uchar g_max = 0;
            uchar r_max = 0;
            for (int ky = 0; ky < size; ky++){
                for (int kx = 0; kx < size; kx++){
                    b_max = std::max(src.at<cv::Vec3b>(y+ky, x+kx)[0], b_max);
                    g_max = std::max(src.at<cv::Vec3b>(y+ky, x+kx)[1], g_max);
                    r_max = std::max(src.at<cv::Vec3b>(y+ky, x+kx)[2], r_max);
                }
            }
            for (int ky = 0; ky < size; ky++){
                for (int kx = 0; kx < size; kx++){
                    dst.at<cv::Vec3b>(y+ky, x+kx)[0] = b_max;
                    dst.at<cv::Vec3b>(y+ky, x+kx)[1] = g_max;
                    dst.at<cv::Vec3b>(y+ky, x+kx)[2] = r_max;
                }
            }
        }
    }

    return dst;
}

cv::Mat gaussianBlur(cv::Mat src)
{
    int size = 3;
    cv::Mat kernel = (cv::Mat_<float>(size, size) << 1, 2, 1, 2, 4, 2, 1, 2, 1)/16.0;

    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; y++){
        for (int x = 0; x < src.cols; x++){
            float b = 0; 
            float g = 0; 
            float r = 0;
            for (int ky = 0; ky < size; ky++){
                for (int kx = 0; kx < size; kx++){
                    if((0 <= y + (ky-size/2)) && (y + (ky-size/2) <= src.rows) && (0 <= x + (kx-size/2)) && (x + (kx-size/2) <= src.cols)){
                        b += (float)src.at<cv::Vec3b>(y + (ky-size/2), x + (kx-size/2))[0] * kernel.at<float>(ky, kx);
                        g += (float)src.at<cv::Vec3b>(y + (ky-size/2), x + (kx-size/2))[1] * kernel.at<float>(ky, kx);
                        r += (float)src.at<cv::Vec3b>(y + (ky-size/2), x + (kx-size/2))[2] * kernel.at<float>(ky, kx);
                    }
                }
            }
            dst.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(b);
            dst.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(g);
            dst.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(r);
        }
    }

    return dst;

}

cv::Mat medianBlur(cv::Mat src)
{
    int size = 3;
    cv::Mat kernel = (cv::Mat_<float>(size, size) << 1, 2, 1, 2, 4, 2, 1, 2, 1)/16.0;

    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int y = 0; y < src.rows; y++){
        for (int x = 0; x < src.cols; x++){
            std::vector<uchar> b_vec, g_vec, r_vec; 
            for (int ky = 0; ky < size; ky++){
                for (int kx = 0; kx < size; kx++){
                    if((0 <= y + (ky-size/2)) && (y + (ky-size/2) <= src.rows) && (0 <= x + (kx-size/2)) && (x + (kx-size/2) <= src.cols)){
                        b_vec.push_back(src.at<cv::Vec3b>(y + (ky-size/2), x + (kx-size/2))[0]);
                        g_vec.push_back(src.at<cv::Vec3b>(y + (ky-size/2), x + (kx-size/2))[1]);
                        r_vec.push_back(src.at<cv::Vec3b>(y + (ky-size/2), x + (kx-size/2))[2]);
                    }else{
                        b_vec.push_back(0);
                        g_vec.push_back(0);
                        r_vec.push_back(0);
                    }
                }
            }
            std::sort(b_vec.begin(), b_vec.end());
            std::sort(g_vec.begin(), g_vec.end());
            std::sort(r_vec.begin(), r_vec.end());
            int median_idx = size * size / 2 + 1;
            dst.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(b_vec[median_idx]);
            dst.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(g_vec[median_idx]);
            dst.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(r_vec[median_idx]);
        }
    }

    return dst;

}