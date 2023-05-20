
#include "opencv2/core.hpp"
//#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/ximgproc/edge_filter.hpp"

#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace std::chrono;
using namespace cv;
using namespace std;

const Mat KNB(Mat &mw, int K);

static void help(const char ** argv)
{
    printf("Usage:\n %s [image_name]\n",argv[0]);
}

const char* keys =
{
    "{help h usage ? |      |   }"
    "{@image1        |   fuente.png  | image1 for process   }"
};

void DarkChannel(Mat &img, int sz, Mat &dst);
const Scalar AtmLight(Mat &img, Mat &dark);
void TransmissionEstimate(Mat &im, Scalar A, int sz, Mat &dst);
void TransmissionRefine(Mat &im, Mat &et);
void Guidedfilter(Mat &im_grey, Mat &transmission_map, int r, float eps);
void Recover(Mat &im, Mat &t, Mat &dst, Scalar A, int tx);

void DarkChannel(Mat &img, int sz, Mat &dst)
{
#if 0
    // Split image into b, g, r channels
    vector<Mat> channels(3);
    cv::split(img, channels);
    
    // Obtaining dark channel, so each pixel's min value
    Mat dc = cv::min(cv::min(channels[0], channels[1]), channels[2]);
    
    // 'erode' image, so calculate the minimun value in the window given by sz
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(sz,sz));
    
    cv::erode(dc, dst, kernel);
#else
#   if 0
    dst = Mat::zeros(img.rows, img.cols, CV_8UC1);
    
    // Reduce memory
    for(int row = 0; row < img.rows; row++)
    {
        for(int col = 0; col < img.cols; col++)
        {
            dst.at<uchar>(row,col) = cv::min(cv::min(img.at<Vec3b>(row,col)[0], img.at<Vec3b>(row,col)[1]), img.at<Vec3b>(row,col)[2]);
        }
    }
#   else
    
    // Init destination and temp Mat
    dst = Mat::zeros(img.rows, img.cols, CV_8UC1);
    Mat temp(img.rows, img.cols, CV_8UC1);
    
    // Move ch0 to dst and ch1 to temp to compare later
    int from_to[] = { 0,0, 1,1 };
    vector <Mat> out { dst, temp };
    
    // Actually move arrays
    cv::mixChannels( &img, 1, &out[0], 2, from_to, 2 );
    
    // Get min from ch0 and ch1
    dst = cv::min(dst, temp);
    
    // dst contains min from ch0 and ch1, need to extract ch2
    int from_to2[]= { 2,0 };
    cv::mixChannels( &img, 1, &temp, 1, from_to2, 1 );
    
    // Get min from min and ch2
    dst = cv::min(dst, temp);
    
    // release temp memory
    temp.release();
    
#   endif
    // 'erode' image, so calculate the minimun value in the window given by sz
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(sz,sz));
    
    cv::erode(dst, dst, kernel, Point((int)(sz/2),(int)(sz/2)), 1, cv::BORDER_CONSTANT);
    
    
#endif
}

const Scalar AtmLight(Mat &im, Mat &dark)
{
    int _rows = im.rows;
    int _cols = im.cols;
    
    int imsz = _rows*_cols;
    
    int numpx = (int)MAX(imsz/1000, 1);
    
    Mat darkvec = dark.reshape(0, 1);
    Mat imvec = im.reshape(0, 1);
    
    Mat indices;
    cv::sortIdx(darkvec, indices, SORT_DESCENDING);
    
    Scalar atmsum(0, 0, 0, 0);
    for(int ind = 0; ind < numpx; ind++)
    {
        atmsum.val[0] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[0];
        atmsum.val[1] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[1];
        atmsum.val[2] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[2];
    }
    
    atmsum.val[0] = atmsum.val[0]/ numpx;
    atmsum.val[1] = atmsum.val[1]/ numpx;
    atmsum.val[2] = atmsum.val[2]/ numpx;
    
    cout << "atmsum: " << atmsum << endl;
    
    return atmsum;
}

void TransmissionEstimate(Mat &im, Scalar A, int sz, Mat &dst)
{
#if 0
    float omega = 0.95;
    
    Mat im3;
    
    vector<Mat> im_ch(3);
    cv::split(im, im_ch);
    
    im_ch[0] = (im_ch[0] / A.val[0]) * 255;
    im_ch[1] = (im_ch[1] / A.val[1]) * 255;
    im_ch[2] = (im_ch[2] / A.val[2]) * 255;
    
    cv::merge(im_ch, im3);
    
    Mat _dark;
    DarkChannel(im3,sz,_dark);
    dst = 255 - omega*_dark;
#else
    float omega = 0.95;
    
    Mat im_airl = Mat::zeros(im.rows, im.cols, CV_8UC3);
    
    // Reduce memory
    for(int row = 0; row < im.rows; row++)
    {
        for(int col = 0; col < im.cols; col++)
        {
            im_airl.at<Vec3b>(row,col)[0] = (im.at<Vec3b>(row,col)[0] / A.val[0]) * 255;
            im_airl.at<Vec3b>(row,col)[1] = (im.at<Vec3b>(row,col)[1] / A.val[1]) * 255;
            im_airl.at<Vec3b>(row,col)[2] = (im.at<Vec3b>(row,col)[2] / A.val[2]) * 255;
        }
    }
    
    Mat _dark;
    DarkChannel(im_airl,sz,_dark);
    dst = 255 - omega*_dark;
    
#endif
}

void TransmissionRefine(Mat &im, Mat &et)
{
    Mat gray;
    cvtColor(im, gray, cv::COLOR_BGR2GRAY);
    
    Guidedfilter(gray, et, 60, 0.0001);
}

void Guidedfilter(Mat &im_grey, Mat &transmission_map, int r, float eps)
{
    // Original
#if 0
    // Conver to float
    im.convertTo(im, CV_32FC3);
    cv::normalize(im, im, 0, 1, cv::NORM_MINMAX);
    
    transmission_map.convertTo(transmission_map, CV_32FC1);
    cv::normalize(transmission_map, transmission_map, 0, 1, cv::NORM_MINMAX);
    
    Mat q, mean_I, mean_p, mean_Ip;
    Mat mean_II, mean_a, mean_b;
    Mat im_p;
    
    cv::boxFilter(im, mean_I, CV_32F, Size(r,r));
    cv::boxFilter(transmission_map, mean_p, CV_32F, Size(r,r));
    cv::boxFilter(im.mul(transmission_map), mean_Ip, CV_32F, Size(r,r));

    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    cv::boxFilter(im.mul(im), mean_II,CV_32F,Size(r,r));
    Mat var_I = mean_II - mean_I.mul(mean_I);
    
    Mat a = cov_Ip/(var_I + eps);
    Mat b = mean_p - a.mul(mean_I);

    cv::boxFilter(a, mean_a, CV_32F, Size(r,r));
    cv::boxFilter(b, mean_b, CV_32F, Size(r,r));
    
    q = im.mul(mean_a) + mean_b;
    
    // Go back to uint8
    q = q * 255;
    q.convertTo(q, CV_8UC1);
    
    return q;

#else
    
    // Reducir memoria
    Mat mean_I = Mat(im_grey.rows, im_grey.cols, CV_32FC1);
    Mat mean_Ip = Mat(im_grey.rows, im_grey.cols, CV_32FC1);
    Mat mean_II = Mat(im_grey.rows, im_grey.cols, CV_32FC1);
    
    // Conver to float
    im_grey.convertTo(im_grey, CV_32FC1);
    im_grey = im_grey/255;
    
    transmission_map.convertTo(transmission_map, CV_32FC1);
    transmission_map = transmission_map/255;
    
    // Mean
    mean_Ip = im_grey.mul(transmission_map);
    
    cv::boxFilter(mean_Ip, mean_Ip, CV_32F, Size(r,r));
    cv::boxFilter(im_grey, mean_I, CV_32F, Size(r,r));
    cv::boxFilter(transmission_map, transmission_map, CV_32F, Size(r,r));
    
    // cov_Ip
    // Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    mean_Ip = mean_Ip - (mean_I).mul(transmission_map);
    
    // Mean
    mean_II = im_grey.mul(im_grey);
    cv::boxFilter(mean_II, mean_II,CV_32F,Size(r,r));
    
    // var_I
    // Mat var_I = mean_II - mean_I.mul(mean_I);
    mean_II = mean_II - (mean_I).mul(mean_I);
    
    // a
    //  Mat a = cov_Ip/(var_I + eps);
    mean_II = cv::max(mean_II, eps);
    mean_Ip = (mean_Ip)/(mean_II);
    // b
    // Mat b = mean_p - a.mul(mean_I);
    mean_I = (mean_Ip).mul(mean_I);
    mean_I = transmission_map - mean_I;
    
    // Mean
    cv::boxFilter(mean_Ip, mean_Ip, CV_32F, Size(r,r));
    cv::boxFilter(mean_I, mean_I, CV_32F, Size(r,r));
    
    mean_Ip = im_grey.mul(mean_Ip);
    transmission_map = mean_Ip + mean_I;
    
    // Go back to uint8
    transmission_map = transmission_map * 255;
    transmission_map.convertTo(transmission_map, CV_8UC1);
    
    
#endif
    
}

void Recover(Mat &im, Mat &t, Mat &dst, Scalar A, int tx)
{
    dst = Mat::zeros(im.rows, im.cols, im.type());
    
#if 0
    int from_to[2];
    Mat temp(im.rows, im.cols, CV_32FC1);
    im.convertTo(im, CV_32FC3);
    
    
    cv::subtract(im, A, dst);
    
    for(int i = 0; i < 3; i++)
    {
        from_to[0] = i;
        from_to[1] = 0;
        cv::mixChannels( &dst, 1, &temp, 1, from_to, 1);
        
        //    temp = (temp/t0*255)
        cv::divide(temp,t,temp,255.f, temp.type());
        
        from_to[0] = 0;
        from_to[1] = i;
        cv::mixChannels( &temp, 1, &dst, 1, from_to, 1);
    }
   
    temp.release();
    cv::add(dst, A, dst);
    dst.convertTo(dst, CV_8UC3);
    
#else
    for(int _row = 0; _row < dst.rows; _row++)
    {
        for(int _col = 0; _col < dst.cols; _col++)
        {
            float factor = 255.f/t.at<uchar>(_row, _col);
            
            int temp  = (im.at<Vec3b>(_row, _col)[0] - A.val[0])*factor + A.val[0];
            dst.at<Vec3b>(_row, _col)[0] = (im.at<Vec3b>(_row, _col)[0] - A.val[0])*factor + A.val[0];
            dst.at<Vec3b>(_row, _col)[1] = (im.at<Vec3b>(_row, _col)[1] - A.val[1])*factor + A.val[1];
            dst.at<Vec3b>(_row, _col)[2] = (im.at<Vec3b>(_row, _col)[2] - A.val[2])*factor + A.val[2];
        }
    }
#endif
    
}

void mat_split(Mat &src, Mat &top, Mat &bot)
{
    top = src(Range(0,src.rows/2), Range(0,src.cols));
    bot = src(Range(src.rows/2,src.rows), Range(0,src.cols));
}

// #include "raw_image.hpp"
//#define PARALLELIZE

int main(int argc, const char ** argv)
{
 
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    if (!parser.has("@image1"))
    {
        help(argv);
        return 0;
    }
    
    // Read image
    string filename = parser.get<string>(0);
    
    Mat I = imread(filename, IMREAD_COLOR);
    if(I.empty())
    {
        cout << "Error loading file" << endl;
    }
    
    
    cout << "I rows: " << I.rows << " :: cols: " << I.cols << endl;
    
    auto start = high_resolution_clock::now();
    
    Mat dark_T, dark_B, I_T, I_B, te_T, te_B, J_T, J_B;
    Mat te(I.rows, I.cols, CV_8UC1);
    Mat dark(I.rows, I.cols, CV_8UC1);
    Mat J(I.rows, I.cols, I.type());
    
    mat_split(I, I_T, I_B);
    mat_split(J, J_T, J_B);
    mat_split(dark, dark_T, dark_B);
    mat_split(te, te_T, te_B);
    
#if defined(PARALLELIZE)
    DarkChannel(I_T, 15, dark_T);
    DarkChannel(I_B, 15, dark_B);
#else
    DarkChannel(I, 15, dark);
#endif
    
//    imshow("I", I);
//    imshow("dark", dark);
//    waitKey(0);
//    return 0 ;
    
    auto _darkchannel = high_resolution_clock::now();
    
    Scalar A   = AtmLight(I,dark);
    Scalar A_T = AtmLight(I_T,dark_T);
    Scalar A_B = AtmLight(I_B,dark_B);
    
    Scalar A_M;
    A_M.val[0] = cv::max(A_T.val[0], A_B.val[0]);
    A_M.val[1] = cv::max(A_T.val[1], A_B.val[1]);
    A_M.val[2] = cv::max(A_T.val[2], A_B.val[2]);
    
    cout << A_T << endl;
    cout << A_B << endl;
    cout << A_M << endl;
    cout << A << endl;
    
    auto _airlight = high_resolution_clock::now();
    
#if defined(PARALLELIZE)
    TransmissionEstimate(I_T, A, 15, te_T);
    TransmissionEstimate(I_B, A, 15, te_B);
#else
    TransmissionEstimate(I, A, 15, te);
#endif
//    imshow("t estimate", te);
    
    auto _transmision = high_resolution_clock::now();
    
#if defined(PARALLELIZE)
    
    
    TransmissionRefine(I_T, te_T);
    TransmissionRefine(I_B, te_B);
    
#else
    TransmissionRefine(I,te);
#endif
    
//    imshow("t refine", te);
    
    auto _transmision_refine = high_resolution_clock::now();
    
#if defined(PARALLELIZE)
    Recover(I_T, te_T, J_T, A, 1);
    Recover(I_B, te_B, J_B, A, 1);
#else
    Recover(I, te, J, A, 1);
#endif
    
    auto stop = high_resolution_clock::now();
    
    auto duration_dc = duration_cast<milliseconds>(_darkchannel - start);
    auto duration_air = duration_cast<milliseconds>(_airlight - _darkchannel);
    auto duration_trams = duration_cast<milliseconds>(_transmision - _airlight);
    auto duration_tramsred = duration_cast<milliseconds>(_transmision_refine - _transmision);
    auto duration_stop = duration_cast<milliseconds>(stop - _transmision_refine);
    auto duration = duration_cast<milliseconds>(stop - start);
    
    cout << "Time taken by Dark channel:        "  << duration_dc.count() << " milliseconds" << endl;
    cout << "Time taken by airlight:            "  << duration_air.count() << " milliseconds" << endl;
    cout << "Time taken by transmision estimate:"  << duration_trams.count() << " milliseconds" << endl;
    cout << "Time taken by tranmsmision refined:"  << duration_tramsred.count() << " milliseconds" << endl;
    cout << "Time taken by recover :            "  << duration_stop.count() << " milliseconds" << endl;
    cout << "Time total:                        "  << duration.count() << " milliseconds" << endl;
    
    imshow("I", I);
//    imshow("dark", dark);
    imshow("J", J);
    waitKey(0);
    
    return 0;
}
