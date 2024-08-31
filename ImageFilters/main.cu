#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

const char* CURRENT_FILTER =
    //"Box Blur";
    "Gaussian Blur";
    //"Grayscale";
    //"Invert";
    //"Red Channel";
    //"Threshold Edge Detection";
const char* IMAGE_PATH = "image.jpg";
const int KEY_P = 112;

// Color + 2D index, image[channel][i][j], to 1D
__device__ __host__ int get1dIdx(int width, int channels, int channel, int i, int j) {
    return j * width * channels + i * channels + channel;
}

__global__ void boxBlurKernel(unsigned long* pre, unsigned char* target, int width, int height, int channels, int blur_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        for (int ch = 0;ch < channels;ch++) {
            // Sum of the box, from (x1,y1) to (x2,y2) = pre[x2][y2] - pre[x1-1][y2] - pre[x2][y1-1] + pre[x1-1][y1-1]
            int x1 = max(x - blur_radius + 1, 0),
                y1 = max(y - blur_radius + 1, 0),
                x2 = min(x + blur_radius - 1, width - 1),
                y2 = min(y + blur_radius - 1, height - 1);
            long chSum = pre[get1dIdx(width, channels, ch, x2, y2)];
            if (x1 > 0) {
                // Underflow?
                chSum -= pre[get1dIdx(width, channels, ch, x1-1, y2)];
            }

            if (y1 > 0) {
                chSum -= pre[get1dIdx(width, channels, ch, x2, y1-1)];
            }

            if (x1 > 0 && y1 > 0) {
                chSum += pre[get1dIdx(width, channels, ch, x1 - 1, y1 - 1)];
            }

            int boxSide = blur_radius * 2 - 1,
                boxPixels = boxSide * boxSide,
                avg = chSum / boxPixels;
            target[get1dIdx(width, channels, ch, x, y)] = avg;
        }
    }
}

__global__ void gaussianBlurKernel(float* kernel, unsigned char* source, unsigned char* target, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int totalSize = width * height * channels;
        for (int ch = 0;ch < channels;ch++) {
            float weightedSum = 0;
            for (int i = -2;i <= 2;i++) {
                for (int j = -2;j <= 2;j++) {
                    int idx = (y + j) * width * channels + (x + i) * channels;
                    idx += ch;
                    if (idx >= 0 && idx < totalSize) {
                        // Get the flat index + move indices from [-2,2] to [0,4] for the kernel
                        int kernelIdx = (i + 2) * 5 + (j + 2);
                        //printf("%d,%d kernel value: %d\n", i, j, kernel[kernelIdx]);
                        weightedSum += (int)source[idx] * kernel[kernelIdx];
                    }
                }
            }

            int centerIdx = y * width * channels + x * channels;
            target[centerIdx + ch] = weightedSum;
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

__global__ void thresholdEdgeDetectionKernel(unsigned char* source, unsigned char* target, int width, int height, int channels, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int currIdx = get1dIdx(width, channels, 0, x, y),
            leftIdx = get1dIdx(width, channels, 0, x, max(y - 1, 0)),
            rightIdx = get1dIdx(width, channels, 0, x, min(y + 1, height - 1)),
            topIdx = get1dIdx(width, channels, 0, max(x - 1, 0), y),
            bottomIdx = get1dIdx(width, channels, 0, min(x + 1, width - 1), y);

        int newIntensity = 0;
        if (abs(source[currIdx] - source[leftIdx]) > threshold ||
            abs(source[currIdx] - source[rightIdx]) > threshold ||
            abs(source[currIdx] - source[topIdx]) > threshold ||
            abs(source[currIdx] - source[bottomIdx]) > threshold) {
            newIntensity = 255;
        }

        for (int ch = 0;ch < channels;ch++) {
            target[currIdx + ch] = newIntensity;
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

    cv::namedWindow(CURRENT_FILTER, cv::WINDOW_AUTOSIZE);

    bool kernelExecuted = true,
        waitForKeyOutside = true; // Don't wait twice, some filters handle it themselves

    if (strcmp(CURRENT_FILTER, "Box Blur") == 0) {
        waitForKeyOutside = false;

        // Prefix sum of a matrix. sum[i][j] = val[i][j] + sum[i-1][j] + sum[i][j-1] - sum[i-1][j-1]
        unsigned long* pre = (unsigned long*)malloc(imageDataSize * sizeof(unsigned long));
        for (int ch = 0;ch < channels;ch++) {
            for (int i = 0;i < width;i++) {
                for (int j = 0;j < height;j++) {
                    int chValue = image.at<cv::Vec3b>(j, i)[ch];
                    int idx = get1dIdx(width, channels, ch, i, j);
                    pre[idx] = chValue;
                    if (i > 0) {
                        int topIdx = get1dIdx(width, channels, ch, i-1, j);
                        pre[idx] += pre[topIdx];
                    }

                    if (j > 0) {
                        int leftIdx = get1dIdx(width, channels, ch, i, j-1);
                        pre[idx] += pre[leftIdx];
                    }

                    if (i > 0 && j > 0) {
                        int topLeftIdx = get1dIdx(width, channels, ch, i-1, j-1);
                        pre[idx] -= pre[topLeftIdx];
                    }
                }
            }
            
        }

        unsigned long* devicePre;
        cudaMalloc(&devicePre, imageDataSize * sizeof(unsigned long));

        cudaMemcpy(devicePre, pre, imageDataSize * sizeof(unsigned long), cudaMemcpyHostToDevice);

        cudaEvent_t startEvent, endEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&endEvent);

        printf("Press P to pause\n");

        for (int radius = 2,direction = 1;;radius+=direction) {
            cudaEventRecord(startEvent, 0);
            //TODO deviceImageData is overwritten?
            boxBlurKernel << <gridSize, blockSize >> > (devicePre, deviceImageData, width, height, channels, radius);
            cudaEventRecord(endEvent, 0);
            
            cudaEventSynchronize(endEvent);

            float elapsedTime;
            cudaEventElapsedTime(&elapsedTime, startEvent, endEvent);

            printf("Radius %d,\texecuted in %fms\n", radius, elapsedTime);

            hostImageData = (unsigned char*)malloc(imageDataSize);
            cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);

            cv::Mat modifiedImage = cv::Mat(height, width, CV_8UC3, hostImageData);
            cv::imshow(CURRENT_FILTER, modifiedImage);
            int key = cv::waitKey(100);

            if (key != -1) {
                printf("Pressed %d\n", key);

                if (key == KEY_P) {
                    waitForKeyOutside = true;
                }
                break;
            }

            if (radius == 50 || radius == 1) {
                direction = -direction;
            }
        }

        cudaEventDestroy(startEvent);
        cudaEventDestroy(endEvent);

        cudaFree(devicePre);

        free(pre);
    }
    else if (strcmp(CURRENT_FILTER, "Gaussian Blur") == 0) {
        float kernel[5][5]{},
            kernelSum = 0.0f,
            pi = 3.14159f,
            sigma = 1.0f;

        for (int x = -2;x <= 2;x++) {
            for (int y = -2;y <= 2;y++) {
                int i = x + 2,
                    j = y + 2;
                kernel[i][j] = 1 / (2 * pi * sigma * sigma) * exp(-((float)x * x + y * y) / 2 * sigma * sigma);
                kernelSum += kernel[i][j];
            }
        }

        // Normalize
        for (int i = 0;i < 5;i++) {
            for (int j = 0;j < 5;j++) {
                kernel[i][j] /= kernelSum;
            }
        }

        printf("Gaussian blur kernel:\n");
        for (int i = 0;i < 5;i++) {
            for (int j = 0;j < 5;j++) {
                printf("%f ", kernel[i][j]);
            }
            printf("\n");
        }

        int kernelDataSize = 5 * 5 * sizeof(float);
        float* deviceKernel;
        cudaMalloc(&deviceKernel, kernelDataSize);
        cudaMemcpy(deviceKernel, (float*)kernel, kernelDataSize, cudaMemcpyHostToDevice);

        unsigned char* deviceImageDataCopy;
        cudaMalloc(&deviceImageDataCopy, imageDataSize);
        cudaMemcpy(deviceImageDataCopy, image.data, imageDataSize, cudaMemcpyHostToDevice);

        gaussianBlurKernel << <gridSize, blockSize >> > (deviceKernel, deviceImageDataCopy, deviceImageData, width, height, channels);

        hostImageData = (unsigned char*)malloc(imageDataSize);
        cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);

        cudaFree(deviceKernel);
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
    else if (strcmp(CURRENT_FILTER, "Threshold Edge Detection") == 0) {
        printf("Applying grayscale\n");
        grayscaleKernel << <gridSize, blockSize >> > (deviceImageData, width, height, channels);

        printf("Applying threshold edge detection\n");
        unsigned char* deviceImageDataCopy;
        cudaMalloc(&deviceImageDataCopy, imageDataSize);
        cudaMemcpy(deviceImageDataCopy, image.data, imageDataSize, cudaMemcpyHostToDevice);

        thresholdEdgeDetectionKernel << <gridSize, blockSize >> > (deviceImageDataCopy, deviceImageData, width, height, channels, 20);

        hostImageData = (unsigned char*)malloc(imageDataSize);
        cudaMemcpy(hostImageData, deviceImageData, imageDataSize, cudaMemcpyDeviceToHost);
    }
    else {
        printf("Error: unknown filter\n");
        kernelExecuted = false;
    }

    if (kernelExecuted) {
        int executions = gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z;
        printf("Kernel was executed %d times, wasted %d\n", executions, executions - (int)image.total());

        cv::Mat modifiedImage = cv::Mat(height, width, CV_8UC3, hostImageData);

        cv::imwrite("modified.bmp", modifiedImage);

        cv::imshow(CURRENT_FILTER, modifiedImage);
    }

    if (waitForKeyOutside) {
        cv::waitKey(0);
    }

    cudaFree(deviceImageData);
    return 0;
}
