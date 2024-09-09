#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <iostream>
#include <cmath>
#include <chrono>

#include "files_helpers_cuda_no_filesystem.h"

#define USE_CUDA
//#define USE_CPU

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

cv::Mat resizeImage(const cv::Mat& image, int scalePercent) {
    if (scalePercent == 100) {
        return image.clone();
    }
    int width = static_cast<int>(image.cols * scalePercent / 100.0);
    int height = static_cast<int>(image.rows * scalePercent / 100.0);
    cv::Size dim(width, height);
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, dim, 0, 0, cv::INTER_LINEAR);
    return resizedImage;
}

std::vector<cv::Point> calcFilterPoints(const std::vector<cv::Point>& points, double threshold) {
    std::vector<cv::Point> filteredPoints;
    for (const auto& point : points) {
        bool isFar = true;
        for (const auto& fp : filteredPoints) {
            if (cv::norm(point - fp) < threshold) {
                isFar = false;
                break;
            }
        }
        if (isFar) {
            filteredPoints.push_back(point);
        }
    }
    return filteredPoints;
}

std::vector<cv::Mat> getAllTemplateImages(const cv::Mat& templateImg) {
    std::vector<cv::Mat> templateImages;
    std::vector<int> scalePercents = { 100, 50, 25 };

    for (int scalePercent : scalePercents) {
        cv::Mat resizedTemplate = resizeImage(templateImg, scalePercent);
        cv::Mat resizedTemplateRotate90, resizedTemplateRotate180, resizedTemplateRotate270;

        cv::transpose(resizedTemplate, resizedTemplateRotate90);
        cv::flip(resizedTemplate, resizedTemplateRotate180, 0);
        cv::transpose(resizedTemplateRotate180, resizedTemplateRotate270);

        templateImages.push_back(resizedTemplate);
        templateImages.push_back(resizedTemplateRotate90);
        templateImages.push_back(resizedTemplateRotate180);
        templateImages.push_back(resizedTemplateRotate270);
    }
    return templateImages;
}

//cv::Mat calcTMSqdiffNormed(const cv::Mat& image, const cv::Mat& templ) {
//    // Ensure the input images are of type CV_64F (double)
//    cv::Mat img, tmpl;
//    image.convertTo(img, CV_64F);
//    templ.convertTo(tmpl, CV_64F);
//
//    int imageHeight = img.rows;
//    int imageWidth = img.cols;
//    int templateHeight = tmpl.rows;
//    int templateWidth = tmpl.cols;
//
//    int resultHeight = imageHeight - templateHeight + 1;
//    int resultWidth = imageWidth - templateWidth + 1;
//
//    cv::Mat result(resultHeight, resultWidth, CV_64F, cv::Scalar(1.0)); // Initialize with the worst value (1.0)
//
//    cv::Mat sumTemplate, sumTemplateSq;
//    cv::integral(tmpl, sumTemplate, sumTemplateSq, CV_64F);
//
//    cv::Mat sumImage, sumImageSq;
//    cv::integral(img, sumImage, sumImageSq, CV_64F);
//
//    double eps = 1e-8;
//
//    for (int i = 0; i < resultHeight; ++i) {
//        for (int j = 0; j < resultWidth; ++j) {
//            cv::Rect region(j, i, templateWidth, templateHeight);
//            cv::Mat imagePatch = img(region);
//
//            double sumImagePatch = sumImage.at<double>(i + templateHeight, j + templateWidth)
//                - sumImage.at<double>(i, j + templateWidth)
//                - sumImage.at<double>(i + templateHeight, j)
//                + sumImage.at<double>(i, j);
//
//            double sumImagePatchSq = sumImageSq.at<double>(i + templateHeight, j + templateWidth)
//                - sumImageSq.at<double>(i, j + templateWidth)
//                - sumImageSq.at<double>(i + templateHeight, j)
//                + sumImageSq.at<double>(i, j);
//
//            double sumTemplateVal = sumTemplate.at<double>(templateHeight, templateWidth);
//            double sumTemplateSqVal = sumTemplateSq.at<double>(templateHeight, templateWidth);
//
//            cv::Mat diff = tmpl - imagePatch;
//            cv::Mat sqrDiff;
//            cv::pow(diff, 2, sqrDiff);
//            double numerator = cv::sum(sqrDiff)[0];
//
//            double denominator = std::sqrt(sumTemplateSqVal * sumImagePatchSq);
//
//            if (denominator > eps) {
//                result.at<double>(i, j) = numerator / denominator;
//            }
//        }
//    }
//
//    return result;
//}

//cv::Mat matchTemplateSqdiffNormed_GPU(const cv::Mat& image, const cv::Mat& templateImg, int templateIndex, int numOfTemplates, const cv::Mat& mask = cv::Mat()) {
//    cv::cuda::GpuMat gpu_result;
//
//    int matrix_type = CV_32FC1;
//
//    cv::cuda::GpuMat gpu_image;
//    gpu_image.upload(image);
//
//    cv::cuda::GpuMat gpu_templateImg;
//    gpu_templateImg.upload(templateImg);
//
//    cv::cuda::GpuMat gpu_mask;
//    gpu_mask.upload(mask);
//
//    cv::cuda::GpuMat gpu_image_float;
//    gpu_image.convertTo(gpu_image_float, matrix_type);
//
//    cv::cuda::GpuMat gpu_templateImg_float;
//    gpu_templateImg.convertTo(gpu_templateImg_float, matrix_type);
//
//    cv::cuda::GpuMat gpu_mask_float;
//    if (!gpu_mask.empty()) {
//        gpu_mask.convertTo(gpu_mask_float, matrix_type);
//    }
//
//    int resultRows = gpu_image_float.rows - gpu_templateImg_float.rows + 1;
//    int resultCols = gpu_image_float.cols - gpu_templateImg_float.cols + 1;
//    gpu_result.create(resultRows, resultCols, matrix_type);
//
//
//
//    cv::cuda::GpuMat sqr_gpu_templateImg_float;
//    cv::cuda::multiply(gpu_templateImg_float, gpu_templateImg_float, sqr_gpu_templateImg_float);
//    float sumSqrTemplate = cv::cuda::sum(sqr_gpu_templateImg_float)[0];
//    cv::cuda::Stream stream;
//
//    cv::Mat cpu_result;
//    cpu_result.create(resultRows, resultCols, matrix_type);
//
//    for (int i = 0; i < resultRows; ++i)
//    {
//        //if (i % 100 == 0)
//        //{
//        std::cout << templateIndex + 1 << " out of " << numOfTemplates << ", " << i + 1 << " out of " << resultRows << std::endl;
//        //}
//
//        for (int j = 0; j < resultCols; ++j) {
//            cv::Rect roi(j, i, gpu_templateImg_float.cols, gpu_templateImg_float.rows);
//            cv::cuda::GpuMat gpu_patch = gpu_image_float(roi);
//            cv::cuda::GpuMat gpu_diff;
//            cv::cuda::subtract(gpu_patch, gpu_templateImg_float, gpu_diff);
//            cv::cuda::GpuMat gpu_mask_0_1;
//            cv::cuda::divide(gpu_mask_float, 255.0, gpu_mask_0_1);
//
//            if (!gpu_mask_float.empty()) {
//                cv::cuda::multiply(gpu_diff, gpu_mask_0_1, gpu_diff);
//            }
//
//            cv::cuda::GpuMat sqr_gpu_diff;
//            cv::cuda::multiply(gpu_diff, gpu_diff, sqr_gpu_diff);
//            float nominator = cv::cuda::sum(sqr_gpu_diff)[0];
//
//            cv::cuda::GpuMat sqr_gpu_patch;
//            cv::cuda::multiply(gpu_patch, gpu_patch, sqr_gpu_patch);
//            float sumSqrPatch = cv::cuda::sum(sqr_gpu_patch)[0];
//            float denominator = std::sqrt(sumSqrTemplate * sumSqrPatch);
//
//            float currentVal = 1.0f;
//            if (denominator > 1e-8) {
//                currentVal = nominator / denominator;
//            }
//            cpu_result.at<float>(i, j) = currentVal;
//        }
//    }
//    std::cout << std::endl;
//
//
//    //gpu_result.download(cpu_result);
//    return cpu_result;
//}

__host__ __device__ void calc_result_matrix(
    const float* image,
    int imageRows,
    int imageCols,
    const float* templateImg,
    int templateRows,
    int templateCols,
    float* result,
    int resultRows,
    int resultCols,
    const float* mask,
    bool useMask,
    float sumSqrTemplate,
    size_t imageStep,
    size_t templateStep,
    size_t resultStep,
    size_t maskStep,
    int x,
    int y
)
{
    if (x < resultCols && y < resultRows)
    {
        float nominator = 0.0f;
        float sum_sqr_template = 0.0f;
        float sum_sqr_patch = 0.0f;
        for (int i = 0; i < templateRows; ++i) {
            for (int j = 0; j < templateCols; ++j) {
                int imageIdx = (y + i) * imageStep + (x + j);
                int templateIdx = i * templateStep + j;                

                float image_val = image[imageIdx];
                float template_val = templateImg[templateIdx];

                float template_patch_diff = image_val - template_val;
                float sqr_image_val = image_val * image_val;  
                float sqr_template_val = template_val * template_val;
                if (useMask) {
                    int maskIdx = i * maskStep + j;
                    float mask_0_1_val = mask[maskIdx] / 255.0f;
                    template_patch_diff *= mask_0_1_val;
                    sqr_image_val *= mask_0_1_val;
                    sqr_template_val *= mask_0_1_val;
                }
                sum_sqr_template += sqr_template_val;
                sum_sqr_patch += sqr_image_val;
                float sqr_template_patch_diff = template_patch_diff * template_patch_diff;
                nominator += sqr_template_patch_diff;
            }
        }

        float denominator = sqrt(sum_sqr_template * sum_sqr_patch);
        float currentVal = (denominator > 1e-8) ? (nominator / denominator) : 1.0f;
        int result_index = y * resultStep + x;
        result[result_index] = currentVal;
    }
}

//__host__ __device__ void calc_result_matrix(
//    const float* image,
//    int imageRows,
//    int imageCols,
//    const float* templateImg,
//    int templateRows,
//    int templateCols,
//    float* result,
//    int resultRows,
//    int resultCols,
//    const float* mask,
//    bool useMask,
//    float sumSqrTemplate,
//    size_t step,
//    int x,
//    int y
//)
//{
//    if (x < resultCols && y < resultRows)
//    {
//        int result_index = y * step + x;
//        printf("result_index = %d\n", result_index);
//        result[result_index] = 0.2;
//    }
//}


// CUDA Kernel
__global__ void matchTemplateKernel(
    const float* image, 
    int imageRows, 
    int imageCols,
    const float* templateImg, 
    int templateRows, 
    int templateCols,
    float* result, 
    int resultRows, 
    int resultCols,
    const float* mask, 
    bool useMask, 
    float sumSqrTemplate,
    size_t imageStep,
    size_t templateStep,
    size_t resultStep,
    size_t maskStep
    ) 
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    calc_result_matrix
    (
        image,
        imageRows,
        imageCols,
        templateImg,
        templateRows,
        templateCols,
        result,
        resultRows,
        resultCols,
        mask,
        useMask,
        sumSqrTemplate,
        imageStep,
        templateStep,
        resultStep,
        maskStep,
        x,
        y
    );
}

void matchTemplateKernel_cpu(
    const float* image,
    int imageRows,
    int imageCols,
    const float* templateImg,
    int templateRows,
    int templateCols,
    float* result,
    int resultRows,
    int resultCols,
    const float* mask,
    bool useMask,
    float sumSqrTemplate,
    size_t imageStep,
    size_t templateStep,
    size_t resultStep,
    size_t maskStep,
    int blockSizeX, 
    int blockSizeY,
    int gridSizeX, 
    int gridSizeY
    ) 
{
    for (int thread_Idx_y = 0; thread_Idx_y < blockSizeY; thread_Idx_y++)
    {
        std::cout << thread_Idx_y + 1 << " out of " << blockSizeY << std::endl;
        for (int thread_Idx_x = 0; thread_Idx_x < blockSizeX; thread_Idx_x++)
        {
            for (int block_Idx_x = 0; block_Idx_x < gridSizeX; block_Idx_x++)
            {
                for (int block_Idx_y = 0; block_Idx_y < gridSizeY; block_Idx_y++)
                {

                    int x = block_Idx_x * blockSizeX + thread_Idx_x;
                    int y = block_Idx_y * blockSizeY + thread_Idx_y;

                    calc_result_matrix
                    (
                        image,
                        imageRows,
                        imageCols,
                        templateImg,
                        templateRows,
                        templateCols,
                        result,
                        resultRows,
                        resultCols,
                        mask,
                        useMask,
                        sumSqrTemplate,
                        imageStep,
                        templateStep,
                        resultStep,
                        maskStep,
                        x,
                        y
                    );
                }
            }            
        }
    }
}

cv::Mat matchTemplateSqdiffNormed(const cv::Mat& grayscaleImage, const cv::Mat& grayscaleTemplate, const cv::Mat& mask = cv::Mat()) 
{
    int matrix_type = CV_32FC1;

    int resultRows = grayscaleImage.rows - grayscaleTemplate.rows + 1;
    int resultCols = grayscaleImage.cols - grayscaleTemplate.cols + 1;

    cv::Mat image_float;
    grayscaleImage.convertTo(image_float, matrix_type);

    cv::Mat templateImg_float;
    grayscaleTemplate.convertTo(templateImg_float, matrix_type);

    cv::cuda::GpuMat d_image_float(image_float);
    cv::cuda::GpuMat d_templateImg_float(templateImg_float);
    cv::cuda::GpuMat d_mask_float;

    cv::Mat mask_float;
    if (!mask.empty()) {
        mask.convertTo(mask_float, matrix_type);
        d_mask_float.upload(mask_float);
    }

    cv::Mat result_cpu(resultRows, resultCols, matrix_type);
    //cv::cuda::GpuMat d_result(result_cpu);
    cv::cuda::GpuMat d_result(resultRows, resultCols, matrix_type);

    cv::cuda::GpuMat d_sqr_templateImg_float;
    cv::cuda::multiply(d_templateImg_float, d_templateImg_float, d_sqr_templateImg_float);
    float sumSqrTemplate = cv::cuda::sum(d_sqr_templateImg_float)[0];

    cv::Mat sqr_templateImg_float;
    cv::cuda::multiply(templateImg_float, templateImg_float, sqr_templateImg_float);
    cv::Scalar sumSqrTemplate_scalar = cv::sum(sqr_templateImg_float);
    float sumSqrTemplate1 = sumSqrTemplate_scalar[0];

    float diff_sumSqrTemplate = sumSqrTemplate1 - sumSqrTemplate;

    int blockSizeX = 16;
    int blockSizeY = 16;
    int gridSizeX = (resultCols + blockSizeX - 1) / blockSizeX;
    int gridSizeY = (resultRows + blockSizeY - 1) / blockSizeY;
    // Launch CUDA kernel

#ifdef USE_CUDA
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize(gridSizeX, gridSizeY);

    size_t gpu_image_step = d_image_float.step / sizeof(float);
    size_t gpu_template_step = d_templateImg_float.step / sizeof(float);
    size_t gpu_result_step = d_result.step / sizeof(float);
    size_t gpu_mask_step = d_mask_float.step / sizeof(float);
    //printf("matchTemplateKernel\n");
    matchTemplateKernel << <gridSize, blockSize >> > 
        (
        d_image_float.ptr<float>(), 
        d_image_float.rows, 
        d_image_float.cols,
        d_templateImg_float.ptr<float>(), 
        d_templateImg_float.rows, 
        d_templateImg_float.cols,
        d_result.ptr<float>(), 
        resultRows, 
        resultCols,
        d_mask_float.empty() ? nullptr : d_mask_float.ptr<float>(), 
        !d_mask_float.empty(), 
        sumSqrTemplate,
        gpu_image_step,
        gpu_template_step,
        gpu_result_step,
        gpu_mask_step
        );

    cudaError_t cudaGetLastErrorStatus = cudaGetLastError();
    checkCudaErrors(cudaGetLastErrorStatus);
    cudaError_t cudaDeviceSynchronizeStatus = cudaDeviceSynchronize();  // Wait for kernel to finish
    checkCudaErrors(cudaDeviceSynchronizeStatus);
    // Download the result to the CPU
    cv::Mat result;
    d_result.download(result);
#endif //USE_CUDA

#ifdef USE_CPU
    printf("\n");
    printf("matchTemplateKernel_cpu\n");
    size_t cpu_image_step = image_float.cols;
    size_t cpu_template_step = templateImg_float.cols;
    size_t cpu_result_step = resultCols;
    size_t cpu_mask_step = mask_float.cols;
    matchTemplateKernel_cpu
    (
        image_float.ptr<float>(),
        image_float.rows,
        image_float.cols,
        templateImg_float.ptr<float>(),
        templateImg_float.rows,
        templateImg_float.cols,
        result_cpu.ptr<float>(),
        resultRows,
        resultCols,
        mask_float.empty() ? nullptr : mask_float.ptr<float>(),
        !mask_float.empty(),
        sumSqrTemplate,
        cpu_image_step,
        cpu_template_step,
        cpu_result_step,
        cpu_mask_step,
        blockSizeX,
        blockSizeY,
        gridSizeX,
        gridSizeY        
    );
#endif //USE_CPU

#if defined(USE_CUDA) && defined(USE_CPU)
    cv::Mat diff_results = result - result_cpu;
#endif

#if defined(USE_CUDA)
    return result;
#else
    return result_cpu;
#endif //USE_CUDA
}



cv::Mat getImageWithIdentifiedTemplates(const cv::Mat& rgbImage, const std::vector<cv::Mat>& templateImages) {
    //std::vector<cv::Mat> templateImages1;
    //templateImages1.push_back(templateImages[2]);

    cv::Mat rgbImageWithIdentifiedTemplates = rgbImage.clone();
    int numOfTemplates = templateImages.size();

    for (int templateIndex = 0; templateIndex < numOfTemplates; ++templateIndex) {
        cv::Mat rgbaTemplate = templateImages[templateIndex];
        cv::Mat alphaChannel;
        cv::extractChannel(rgbaTemplate, alphaChannel, 3);

        cv::Mat alphaChannelOnly0Or255;
        cv::threshold(alphaChannel, alphaChannelOnly0Or255, 0, 255, cv::THRESH_BINARY);

        cv::Mat grayscaleImage;
        cv::cvtColor(rgbImage, grayscaleImage, cv::COLOR_BGR2GRAY);

        cv::Mat rgbTemplate;
        cv::cvtColor(rgbaTemplate, rgbTemplate, cv::COLOR_RGBA2RGB);

        cv::Mat grayscaleTemplate;
        cv::cvtColor(rgbTemplate, grayscaleTemplate, cv::COLOR_BGR2GRAY);

        //cv::Mat result;
        //int match_method = cv::TM_SQDIFF_NORMED;
        //cv::matchTemplate(grayscaleImage, grayscaleTemplate, result, match_method);
        //double threshold = 0.1;

            // Transfer images to the GPU


        cv::Mat result = matchTemplateSqdiffNormed(grayscaleImage, grayscaleTemplate, alphaChannelOnly0Or255);
        double threshold = 0.04;

        //double minVal, maxVal;
        //cv::Point minLoc, maxLoc;
        //cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        //printf("minVal = %.10lf\n", minVal);
        //printf("maxVal = %.10lf\n", maxVal);
        //printf("\n\n\n");

        std::vector<cv::Point> templateStartPoints;
        cv::Mat thresholdResult;
        cv::threshold(result, thresholdResult, threshold, 1.0, cv::THRESH_BINARY_INV);
        cv::findNonZero(thresholdResult, templateStartPoints);

        std::vector<cv::Point> filteredPoints = calcFilterPoints(templateStartPoints, 10.0);

        for (const auto& point : filteredPoints) {
            cv::rectangle(rgbImageWithIdentifiedTemplates, point, point + cv::Point(rgbaTemplate.cols, rgbaTemplate.rows), cv::Scalar(0, 0, 255), 4);
        }
    }
    return rgbImageWithIdentifiedTemplates;
}

cv::Mat createImageWithTemplates(const cv::Mat& rgbImage, const cv::Mat& origRgbaTemplate, const std::vector<cv::Point>& startPoints) {
    std::vector<cv::Mat> templateImages = getAllTemplateImages(origRgbaTemplate);

    cv::Mat rgbaTemplate = templateImages[2];
    cv::Mat blueChannel, greenChannel, redChannel, alphaChannel;
    cv::extractChannel(rgbaTemplate, blueChannel, 0);
    cv::extractChannel(rgbaTemplate, greenChannel, 1);
    cv::extractChannel(rgbaTemplate, redChannel, 2);
    cv::extractChannel(rgbaTemplate, alphaChannel, 3);

    cv::Mat mask = alphaChannel > 0;
    int templateRows = rgbaTemplate.rows;
    int templateCols = rgbaTemplate.cols;
    cv::Mat rgbImageWithTemplate = rgbImage.clone();

    for (const auto& startPoint : startPoints) {
        int startRow = startPoint.y;
        int startColumn = startPoint.x;
        int endRow = startRow + templateRows;
        int endColumn = startColumn + templateCols;

        for (int currentChannel = 0; currentChannel < 3; ++currentChannel) {
            cv::Mat rgbPatch = rgbImageWithTemplate(cv::Rect(startColumn, startRow, templateCols, templateRows)).clone();
            cv::Mat templateCurrentChannel, patchCurrentChannel, patchWithTemplateCurrentChannel;
            cv::extractChannel(rgbaTemplate, templateCurrentChannel, currentChannel);
            cv::extractChannel(rgbPatch, patchCurrentChannel, currentChannel);

            cv::Mat condition;
            cv::compare(mask, 255, condition, cv::CMP_EQ);
            patchCurrentChannel.copyTo(patchWithTemplateCurrentChannel, ~condition);
            //templateCurrentChannel.copyTo(patchWithTemplateCurrentChannel, ~condition);
            templateCurrentChannel.copyTo(patchWithTemplateCurrentChannel, condition);

            std::vector<cv::Mat> channels(3);
            cv::split(rgbPatch, channels);
            channels[currentChannel] = patchWithTemplateCurrentChannel;
            cv::merge(channels, rgbPatch);

            rgbPatch.copyTo(rgbImageWithTemplate(cv::Rect(startColumn, startRow, templateCols, templateRows)));
        }
    }
    return rgbImageWithTemplate;
}

void create_small_images(cv::Mat& rgb_image, cv::Mat& rgba_template_image)
{
    int template_height = 2;
    int template_width = 3;
    cv::Mat template_grayscale = cv::Mat::zeros(template_height, template_width, CV_8UC1);
    for (int i = 0; i < template_height; i++)
    {
        for (int j = 0; j < template_width; j++)
        {
            uchar value = i * template_width + j;
            template_grayscale.at<uchar>(i, j) = value;
        }
    }

    cv::Mat template_alpha_channel = 255 * cv::Mat::ones(template_height, template_width, CV_8UC1);

    std::vector<cv::Mat> template_channels = { template_grayscale, template_grayscale, template_grayscale, template_alpha_channel };
    cv::merge(template_channels, rgba_template_image);

    int image_height = 5;
    int image_width = 8;
    cv::Mat image_grayscale = cv::Mat::zeros(image_height, image_width, CV_8UC1);
    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++)
        {
            uchar value = i * image_width + j;
            image_grayscale.at<uchar>(i, j) = value;
        }
    }


    int start_y = 1;
    int start_x = 2;
    for (int i = start_y; i < template_height + start_y; i++)
    {
        for (int j = start_x; j < template_width + start_x; j++)
        {
            int template_index_i = i - start_y;
            int template_index_j = j - start_x;
            uchar value = template_grayscale.at<uchar>(template_index_i, template_index_j);
            image_grayscale.at<uchar>(i, j) = value;
        }
    }

    std::vector<cv::Mat> image_channels = { image_grayscale, image_grayscale, image_grayscale };
    cv::merge(image_channels, rgb_image);
}

int main() {
    std::string image_name = "iron_dome.jpeg";
    std::string template_name = "smiley160px.png";

    My_files my_file_system;
    std::string base_folder_full_path = my_file_system.get_directory_base_path();
    std::string images_folder_full_path = base_folder_full_path + "\\" + "images";
    std::string image_file_full_path = images_folder_full_path + "\\" + image_name;
    std::string template_file_full_path = images_folder_full_path + "\\" + template_name;

    cv::Mat rgbImageWithTemplates = cv::imread(image_file_full_path, cv::IMREAD_UNCHANGED);
    cv::Mat rgbaTemplate = cv::imread(template_file_full_path, cv::IMREAD_UNCHANGED);
    std::vector<cv::Mat> templateImages = getAllTemplateImages(rgbaTemplate);


    //image_name = "puppy.jpg";
    //image_file_full_path = images_folder_full_path + "\\" + image_name;
    //cv::Mat rgbImage = cv::imread(image_file_full_path, cv::IMREAD_UNCHANGED);
    //std::vector<cv::Point> startPoints = { cv::Point(1500, 1000), cv::Point(200, 400), cv::Point(800, 900) };
    //rgbImageWithTemplates = createImageWithTemplates(rgbImage, rgbaTemplate, startPoints);


    //create_small_images(rgbImageWithTemplates, rgbaTemplate);
    //templateImages = getAllTemplateImages(rgbaTemplate);


    auto start_time = std::chrono::high_resolution_clock::now();
    cv::Mat rgbImageWithIdentifiedTemplates = getImageWithIdentifiedTemplates(rgbImageWithTemplates, templateImages);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "getImageWithIdentifiedTemplates took " << duration.count() << " seconds." << std::endl;
    cv::Mat resizedRgbImageWithIdentifiedTemplates = resizeImage(rgbImageWithIdentifiedTemplates, 50);

    cv::imshow("Detecting smileys", resizedRgbImageWithIdentifiedTemplates);
    cv::waitKey(0);

    return 0;
}
