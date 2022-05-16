#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
void histogramMatching(Mat & reference, Mat & input, Mat & result) {
        const float HISMATCH = 0.001;
        double min, max;

        vector<Mat> reference_channels;
        split(reference, reference_channels);

        vector<Mat> input_channels;
        split(input, input_channels);

        int histSize = 256;

        float range[] = { 0,256 };
        const float* histrange = { range };

        bool uniform = true;

        for (int i = 0; i < 3; i++) {
            Mat reference_histogram, input_histogram;
            Mat reference_histogram_accum, input_histogram_accum;
            calcHist(&input_channels[i], 1, 0, Mat(), input_histogram, 1, &histSize, &histrange, &uniform);

            try {
                calcHist(&reference_channels[i], 1, 0, Mat(), reference_histogram, 1, &histSize, &histrange, &uniform);
            }
            catch (int n) {
                cout << "The first element is " << n << endl;
            }




            minMaxLoc(reference_histogram, &min, &max);

            normalize(reference_histogram, reference_histogram, min / max, NORM_MINMAX);

            minMaxLoc(input_histogram, &min, &max);
            normalize(input_histogram, input_histogram, min / max, NORM_MINMAX);

            reference_histogram.copyTo(reference_histogram_accum);
            input_histogram.copyTo(input_histogram_accum);

            float* src_cdf_data = input_histogram_accum.ptr<float>();
            float* dst_cdf_data = reference_histogram_accum.ptr<float>();

            for (int j = 1; j < 256; j++) {
                src_cdf_data[j] += src_cdf_data[j - 1];
                dst_cdf_data[j] += dst_cdf_data[j - 1];
            }

            minMaxLoc(reference_histogram_accum, &min, &max);
            normalize(reference_histogram_accum, reference_histogram_accum, min / max, 1.0, NORM_MINMAX);
            minMaxLoc(input_histogram_accum, &min, &max);
            normalize(input_histogram_accum, input_histogram_accum, min / max, 1.0, NORM_MINMAX);

            //BEGIN Matching
            Mat lut(1, 256, CV_8UC1);
            uchar* M = lut.ptr<uchar>();
            uchar last = 0;
            for (int j = 0; j < input_histogram_accum.rows; j++) {
                float F1 = dst_cdf_data[j];
                int i = 0;
                for (uchar k = last; k < reference_histogram_accum.rows; k++) {
                    i++;
                    float F2 = src_cdf_data[k];
                    if (abs(F2 - F1) < HISMATCH || F2 > F1) {
                        M[j] = k;
                        last = k;
                        break;
                    }
                }

            }

            LUT(input_channels[i], lut, input_channels[i]);

        }
        merge(input_channels, result);
    }

int main() {
    Mat input = imread("Lenna_gray.png", IMREAD_COLOR);
    if (input.empty()) {
        cout << "Image is empty" << endl;
        return -1;
    }
    Mat reference = imread("Moon.png", IMREAD_COLOR);
    if (reference.empty()) {
        cout << "Reference Image is empty" << endl;
        return -1;
    }

    Mat result = input.clone();

    namedWindow("Reference", WINDOW_AUTOSIZE);
    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Result", WINDOW_AUTOSIZE);
    imshow("Reference", reference);
    imshow("Input", input);
    histogramMatching(reference, input, result);
    imshow("Result", result);
    waitKey(0);
    return 0;
}
