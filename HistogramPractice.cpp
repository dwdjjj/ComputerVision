#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int moon_size = 0;
// 이미지 히스토그램 구하기
void imgHist(Mat img, int hist[]) {
    for (int i=0; i<256; i++) {
        hist[i] = 0;
    }
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            hist[img.at<uchar>(i, j)]++;
        }
    }
    
}
// Pd(확률분포) 구하기
void imgProbabilityDistribution(int hist[], double hist_Pd[], int imgSize) {
    for (int i=0; i<256; i++) {
        hist_Pd[i] = (double)hist[i] / imgSize;
    }
}
// 누적 히스토그램
void cumulativeHist(int hist[], int cum_hist[]) {
    int sum = 0;
    for (int i=0; i<256; i++) {
        sum += hist[i];
        cum_hist[i] = sum;
    }
}
// 정규화
void normalizingHist(int cumHist[], int normalHist[], int imgSize) {
    for (int i=0; i<256; i++) {
        normalHist[i] = round((double)cumHist[i] / imgSize * 255);
    }
}

void get_Match(Mat img, Mat outimg, int X_Size, int Y_Size, int* moon_hist) {
    int i, j, tmp = 0;
    int histogramMatch[256];
    
    printf("Start HistoGram Specification \n");

    for (i = 0; i < 256; i++) {
        histogramMatch[i] = 0;
        for (j = 0; j < 256; j++) {
            if ((i - moon_hist[j]) > 0) {
                histogramMatch[i] = j;
            }
        }
    }

    for (i = 0; i < Y_Size; ++i) {
        for (j = 0; j < X_Size; ++j) {
            outimg.at<uchar>(i, j) = histogramMatch[img.at<uchar>(i, j)];
        }
    }

//    for (i = 0; i < 256; i++)
//        printf("result histomatch [%d] : %d\n", i, histogramMatch[i]);

}

int main() {
    Mat moon = imread("Moon.png", IMREAD_GRAYSCALE);
    Mat lena = imread("Lenna_gray.png", IMREAD_GRAYSCALE);
    if (moon.empty() || lena.empty()) {
        cout << "can not open or find the image file" << "\n";
        return -1;
    }

    CV_Assert(moon.depth() == CV_8U);
    
    int moon_size = moon.rows * moon.cols;
    int lena_size = lena.rows * lena.cols;
//    cout << moon_size << " " << lena_size;
    
    int moon_hist[256] = {0, }, lena_hist[256] = {0, };
    imgHist(moon, moon_hist);   imgHist(lena, lena_hist);
//    cout << "Moon_hist" << "\n";
//    for (int i=0; i<256; i++) {
//        cout << "Ns" << i << " = " << moon_hist[i] << " ";
//    }
//    cout << "확률분포";
    double moon_Pd[256], lena_Pd[256];
    
    imgProbabilityDistribution(moon_hist, moon_Pd, moon_size);
    imgProbabilityDistribution(lena_hist, lena_Pd, lena_size);
    
//    for (int i=0; i<256; i++) {
//        cout << moon_Pd[i] << " ";
//    }
//    cout << "\n";
    
    int cum_moon_hist[256], cum_lena_hist[256];
    cumulativeHist(moon_hist, cum_moon_hist);
    cumulativeHist(lena_hist, cum_lena_hist);

//    for (int i=0; i<256; i++) {
//        cout << cum_moon_hist[i] << " ";
//    }
//    cout << "\n";
    
    cout << "누적합구해두고 픽셀전체 개수로 나누고 255곱해줌";
    int normalized_moon[256], normalized_lena[256];
    normalizingHist(cum_moon_hist, normalized_moon, moon_size);
    normalizingHist(cum_lena_hist, normalized_lena, lena_size);
    for (int i=0; i<256; i++) {
//        normalized_moon[i] = round((double)cum_moon_hist[i] / moon_size * 255);
//        normalized_lena[i] = round((double)cum_lena_hist[i] / lena_size * 255);
        cout << normalized_moon[i] << " ";
    }
    cout << "\n";
    for (int i=0; i<256; i++) {
        cout << "Z" << i << "값: " << normalized_lena[i] << "\n";
    }
    
    cout << "equalization";
    double equal_moon[256] = {0, }, equal_lena[256] = {0, };
    for (int i=0; i<256; i++) {
        equal_moon[normalized_moon[i]] += moon_Pd[i];
        equal_lena[normalized_lena[i]] += lena_Pd[i];
        cout << equal_lena[i] << " ";
    }
    
    cout << "\n";
    
    cout << "이퀄라이제이션 값에 사이즈만큼 곱하고";
    int res_moon[256], res_lena[256];
    int sum = 0;
    for (int i=0; i<256; i++) {
        res_moon[i] = round(equal_moon[i] * moon_size);
        res_lena[i] = round(equal_lena[i] * lena_size);
        sum += res_moon[i];
    }
    cout << sum << "\n";
    
    Mat after_moon = moon.clone();
    Mat after_lena = lena.clone();
    for (int i=0; i<moon.rows; i++) {
        for (int j=0; j<moon.cols; j++) {
            after_moon.at<uchar>(i, j) = (normalized_moon[moon.at<uchar>(i, j)]);
        }
    }
    
    for (int i=0; i<lena.rows; i++) {
        for (int j=0; j<lena.cols; j++) {
            after_lena.at<uchar>(i, j) = (normalized_lena[lena.at<uchar>(i, j)]);
        }
    }
    
    // histogram matching
    // 그 이미지에 있는 모든 intensity값(s)들을 z로 가는 맵핑을 통해 inverse없이 구할 수 있음.
    // equalization처럼 히스토그램 구하고, 누적 빈도 수 계산하고, 정규화한 후
    // 기존 화소를 변환하면된다.
    
    int s_to_z_table[256] = {0, };
    int tmp = 0;
    for (int i=0; i<256; i++) {
        for (int j=0; j<256; j++) {
            if(normalized_moon[i] == normalized_lena[j]) {
                s_to_z_table[i] = j;
                tmp = j;
            }
//            else {
//                s_to_z_table[i] = tmp;
//                break;
//            }
        }
    }
    for (int i=0; i<256; i++) {
        cout << s_to_z_table[i] << " ";
    }
    Mat matched_img = lena.clone();
    
    int cumMatched[256] = {0, }, normalizedMatched[256] = {0, };
    cumulativeHist(s_to_z_table, cumMatched);
    normalizingHist(cumMatched, normalizedMatched, lena_size);
    
    for (int i=0; i<lena.rows; i++) {
        for (int j=0; j<lena.cols; j++) {
            matched_img.at<uchar>(i, j) = (normalizedMatched[lena.at<uchar>(i, j)]);
        }
    }
    
//    get_Match(lena, matched_img, lena.rows, lena.cols, moon_hist);

//    for (int i=0; i<matched_img.rows; i++) {
//        for (int j=0; j<matched_img.cols; j++) {
//            matched_img.at <uchar>(i, j) = histM[lena.at<uchar>(i, j)];
//        }
//    }
    
    namedWindow("Display original moon", WINDOW_AUTOSIZE);
    imshow("Display original moon", moon);
    namedWindow("Display equalized moon", WINDOW_AUTOSIZE);
    imshow("Display equalized moon", after_moon);
    namedWindow("Display original lena", WINDOW_AUTOSIZE);
    imshow("Display original lena", lena);
    namedWindow("Display equalized lena", WINDOW_AUTOSIZE);
    imshow("Display equalized lena", after_lena);
    imwrite("equalized_moon.png", after_moon);
    imwrite("equalized_lena.png", after_lena);
    
    namedWindow("Display matched lena", WINDOW_AUTOSIZE);
    imshow("Display matched lena", matched_img);
    waitKey(0);
    
    return 0;
}
