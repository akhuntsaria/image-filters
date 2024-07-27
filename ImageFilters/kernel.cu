#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

__global__ void invertKernel(unsigned char* imageData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width * 3 + x * 3;
        imageData[idx] = 255 - (int)imageData[idx];
        imageData[idx + 1] = 255 - (int)imageData[idx + 1];
        imageData[idx + 2] = 255 - (int)imageData[idx + 2];
    }
}

int main(int argc, char** argv)
{
    cv::Mat image = cv::imread("sample1.png", cv::IMREAD_ANYCOLOR);

    if (image.empty())
    {
        printf("Could not open or find the image\n");
        return -1;
    }

    int width = image.cols,
        height = image.rows,
        channels = image.channels();

    printf("Width: %d, height: %d, channels: %d\n", width, height, channels);

    printf("First few pixels:\n");
    for (int i = 0;i < 10;i++) {
        printf("(%d,%d,%d)\n", (int) image.data[i*3], (int) image.data[i*3+1], (int) image.data[i*3+2]);
    }

    printf("Allocating and copyng %d * %d  bytes to the device\n", (int) image.total(), (int) image.channels());
    
    size_t imageDataSize = image.total() * image.channels();
    unsigned char* deviceImageData;
    cudaMalloc(&deviceImageData, imageDataSize);
    cudaMemcpy(deviceImageData, image.data, imageDataSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);
    invertKernel<<<gridSize, blockSize>>>(deviceImageData, width, height);

    unsigned char* hostImageData = (unsigned char*)malloc(imageDataSize);
    cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);

    cv::Mat invertedImage = cv::Mat(height, width, CV_8UC3, hostImageData);

    cv::namedWindow("Regular", cv::WINDOW_AUTOSIZE);
    cv::imshow("Regular", image);

    cv::namedWindow("Inverted", cv::WINDOW_AUTOSIZE);
    cv::imshow("Inverted", invertedImage);

    cv::waitKey(0);
    return 0;
}
