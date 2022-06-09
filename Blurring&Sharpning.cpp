#include <opencv2/highgui.hpp>
#include <iostream>
const int winSize = 3 * 3;
using namespace cv;
using namespace std;

void insertion_sort(int window[], int n)
{
    int key, i, j;
    for (i = 0; i < 9; i++)
    {
        key = window[i];
        for (j = i - 1; j >= 0 && key < window[j]; j--)
        {
            window[j + 1] = window[j];
        }
        window[j + 1] = key;
    }
}
void medianFilter(Mat &input, Mat &output)
{
    // 슬라이딩 윈도우 3 x 3
    int window[winSize];

    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            output.at<uchar>(i, j) = 0;
        }
    }

    for (int i = 1; i < input.rows - 1; i++)
    {
        for (int j = 1; j < input.cols - 1; j++)
        {
            window[0] = input.at<uchar>(i - 1, j - 1);
            window[1] = input.at<uchar>(i, j - 1);
            window[2] = input.at<uchar>(i + 1, j - 1);
            window[3] = input.at<uchar>(i - 1, j);
            window[4] = input.at<uchar>(i, j);
            window[5] = input.at<uchar>(i + 1, j);
            window[6] = input.at<uchar>(i - 1, j + 1);
            window[7] = input.at<uchar>(i, j + 1);
            window[8] = input.at<uchar>(i + 1, j + 1);

            insertion_sort(window, winSize);    // filter 정렬
            output.at<uchar>(i, j) = window[4]; // 중간값으로 대체
        }
    }
}
void AvereageFilter(Mat &input, Mat &output)
{
    int r = input.rows, c = input.cols;

    int mask[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    for (int i = 1; i < r - 1; i++)
    {
        for (int j = 1; j < c - 1; j++)
        {
            int sum = 0;
            for (int x = 0; x < 3; x++)
            {
                for (int y = 0; y < 3; y++)
                {
                    sum += (input.at<uchar>(i - 1 + x, j - 1 + y) * mask[x][y]);
                }
            }
            output.at<uchar>(i, j) = sum / 9;
        }
    }
}

// 덜 sharp함
void Laplacian(Mat &src, Mat &res)
{
    int r = src.rows, c = src.cols;

    for (int i = 1; i < r - 1; i++)
    {
        for (int j = 1; j < c - 1; j++)
        {
            int sum = 5 * src.at<uchar>(i, j) - src.at<uchar>(i + 1, j) - src.at<uchar>(i - 1, j) - src.at<uchar>(i, j + 1) - src.at<uchar>(i, j - 1);
            if (sum > 255)
                sum = 255;
            if (sum < 0)
                sum = 0;
            res.at<uchar>(i, j) = sum;
        }
    }
}

void Unsharp(Mat &input, Mat &output)
{
    output = input.clone();

    for (int i = 1; i < input.rows - 1; i++)
    {
        uchar *tmp = input.ptr(i - 1);
        uchar *cur = input.ptr(i);
        uchar *next = input.ptr(i + 1);
        uchar *outputX = output.ptr(i);

        for (int j = 1; j < input.cols - 1; j++)
        {
            int sum = 5 * cur[j] - cur[j - 1] - cur[j + 1] - tmp[j] - next[j];
            if (sum > 255)
                sum = 255;
            if (sum < 0)
                sum = 0;
            *outputX++ = sum;
        }
    }
}

int main()
{

    Mat lena = imread("Lenna_gray.png", IMREAD_GRAYSCALE);
    Mat outputImg = lena.clone();
    Mat Out = lena.clone();

    AvereageFilter(lena, Out);
    medianFilter(lena, outputImg);

    Mat sharpImg = lena.clone();
    //    Laplacian(Out, sharpImg);
    Unsharp(lena, sharpImg);

    namedWindow("input lena", WINDOW_AUTOSIZE);
    imshow("input lena", lena);
    namedWindow("output median lena", WINDOW_AUTOSIZE);
    imshow("output median lena", outputImg);
    namedWindow("Output average Img", WINDOW_AUTOSIZE);
    imshow("Output average Img", Out);
    namedWindow("Output sharpening Img", WINDOW_AUTOSIZE);
    imshow("Output sharpening Img", sharpImg);
    imwrite("blurImg.png", Out);
    imwrite("sharpImg.png", sharpImg);

    waitKey(0);

    return 0;
}
