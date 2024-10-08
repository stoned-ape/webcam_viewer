#include "cv_debayer.h"

// #include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/opencv.hpp>

void cv_save(const char *name,int w,int h,void *rgb_data){
    cv::Mat mat2(h, w, CV_16UC4, (uint8_t *)rgb_data, 0);
    assert(rgb_data == mat2.datastart);
    cv::cvtColor(mat2, mat2, cv::COLOR_RGBA2BGRA, 4);
    assert(rgb_data == mat2.datastart);
	cv::imwrite(name, mat2);
}

void cv_bayer_to_rgb(
    const void *bayer_data,void *rgb_data,
    int w,int h){

    uint8_t* byr=(uint8_t*)bayer_data;

    
    // h/=2;
    // for(int i=0;i<w*h;i++){
    //     byr[i]=byr[2*i+1];
    // }

    uint8_t* dbr=(uint8_t*)rgb_data;

    cv::Mat img1(h, w, CV_16UC1, byr, w*2);
	//static 
	cv::Mat img2(h, w, CV_16UC4, dbr, 0);
	const void* old = img2.datastart;
	assert(dbr == img2.datastart);
	cv::cvtColor(img1, img2, cv::COLOR_BayerRG2RGBA, 4);
	assert(old == img2.datastart);
	assert(byr == img1.datastart);


    // cv::Mat bayer_image(h, w, CV_16UC1, (void*)bayer_data);

    // cv::cuda::GpuMat d_bayer_image;
    // d_bayer_image.upload(bayer_image);

    // // Prepare a GpuMat for the debayered output
    // cv::cuda::GpuMat d_rgb_image;

    // // Perform debayering (convert Bayer image to RGB)
    // // Using the BayerBG pattern for this example (change to your Bayer pattern: e.g., BayerRG, BayerGR, BayerGB)
    // cv::cuda::demosaicing(d_bayer_image, d_rgb_image, cv::COLOR_BayerBG2BGR, 3);

    // // Download the debayered image back to CPU memory (if you need to)
    // cv::Mat rgb_image;
    // d_rgb_image.download(rgb_image);
    

}