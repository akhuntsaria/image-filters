#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

const char* CURRENT_FILTER =
    "Box Blur";
    //"Grayscale";
    //"Invert";
    //"Red Channel";
const char* IMAGE_PATH = "image.jpg";

__constant__ float BOX_BLUR_RADIUS = 5;

__global__ void boxBlurKernel(unsigned char* source, unsigned char* target, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int totalSize = width * height * channels;
        for (int ch = 0;ch < channels;ch++) {
            // Sum the channel of pixels in all directions. 9 pixels for radius 2.
            // * * *
            // * p *
            // * * *
            int chSum = 0;
            // For radius 2 the range it [-1,1]
            for (int i = -BOX_BLUR_RADIUS + 1;i < BOX_BLUR_RADIUS;i++) {
                for (int j = -BOX_BLUR_RADIUS + 1;j < BOX_BLUR_RADIUS;j++) {
                    int idx = (y + j) * width * channels + (x + i) * channels;
                    idx += ch;
                    if (idx >= 0 && idx < totalSize) {
                        chSum += (int)source[idx];
                    }
                }
            }
            int boxSide = BOX_BLUR_RADIUS * 2 - 1,
                boxPixels = boxSide * boxSide,
                avg = chSum / boxPixels;

            int centerIdx = y * width * channels + x * channels;
            target[centerIdx + ch] = avg;
        }
    }
}

__global__ void grayscaleKernel(unsigned char* imageData, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width * channels + x * channels;
        int channelAvg = 0;
        for (int ch = 0;ch < channels;ch++) {
            channelAvg += (int)imageData[idx + ch];
        }
        channelAvg /= 3;
        for (int ch = 0;ch < channels;ch++) {
            imageData[idx + ch] = channelAvg;
        }
    }
}

__global__ void invertKernel(unsigned char* imageData, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width * channels + x * channels;
        for (int ch = 0;ch < channels;ch++) {
            imageData[idx + ch] = 255 - (int)imageData[idx + ch];
        }
    }
}

__global__ void redChannelKernel(unsigned char* imageData, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width * channels + x * channels;
        for (int ch = 0;ch < channels;ch++) {
            if (ch == 2) continue;
            imageData[idx + ch] = 0;
        }
    }
}

int main()
{
    cv::Mat image = cv::imread(IMAGE_PATH, cv::IMREAD_ANYCOLOR);

    if (image.empty())
    {
        printf("Could not open or find the image\n");
        return -1;
    }

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", image);

    int width = image.cols,
        height = image.rows,
        channels = image.channels(); // Blue, green, red, etc.

    printf("Width: %d, height: %d, channels: %d\n", width, height, channels);

    printf("First few pixels:\n");
    for (int i = 0;i < 10;i++) {
        printf("(%d,%d,%d)\n", (int) image.data[i*channels], (int) image.data[i*channels+1], (int) image.data[i*channels+2]);
    }

    printf("Allocating and copyng %d * %d  bytes to the device\n", (int) image.total(), channels);
    
    size_t imageDataSize = image.total() * image.channels();
    unsigned char* deviceImageData;
    cudaMalloc(&deviceImageData, imageDataSize);
    cudaMemcpy(deviceImageData, image.data, imageDataSize, cudaMemcpyHostToDevice);

    unsigned char* hostImageData;

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    printf("blockSize: (%d,%d,%d)\n", blockSize.x, blockSize.y, blockSize.z);
    printf("gridSize: (%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z);

    if (strcmp(CURRENT_FILTER, "Box Blur") == 0) {
        unsigned char* deviceImageDataCopy;
        cudaMalloc(&deviceImageDataCopy, imageDataSize);
        cudaMemcpy(deviceImageDataCopy, image.data, imageDataSize, cudaMemcpyHostToDevice);

        boxBlurKernel << <gridSize, blockSize >> > (deviceImageDataCopy, deviceImageData, width, height, channels);

        hostImageData = (unsigned char*)malloc(imageDataSize);
        cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);
    }
    else if (strcmp(CURRENT_FILTER, "Grayscale") == 0) {
        grayscaleKernel << <gridSize, blockSize >> > (deviceImageData, width, height, channels);

        hostImageData = (unsigned char*)malloc(imageDataSize);
        cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);
    }
    else if (strcmp(CURRENT_FILTER, "Invert") == 0) {
        invertKernel << <gridSize, blockSize >> > (deviceImageData, width, height, channels);

        hostImageData = (unsigned char*)malloc(imageDataSize);
        cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);
    }
    else if (strcmp(CURRENT_FILTER, "Red Channel") == 0) {
        redChannelKernel << <gridSize, blockSize >> > (deviceImageData, width, height, channels);

        hostImageData = (unsigned char*)malloc(imageDataSize);
        cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);
    }
    else {
        printf("Unknown filter\n");
        return 1;
    }

    int executions = gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z;
    printf("Kernel was executed %d times, wasted %d\n", executions, executions - image.total());

    cv::Mat modifiedImage = cv::Mat(height, width, CV_8UC3, hostImageData);

    cv::namedWindow(CURRENT_FILTER, cv::WINDOW_AUTOSIZE);
    cv::imshow(CURRENT_FILTER, modifiedImage);

    cv::waitKey(0);
    return 0;
}
