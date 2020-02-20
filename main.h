#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "bmp/EasyBMP.h"
using namespace std;

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

float *readLikeGrayScale(char *filePathInput, unsigned int *rows, unsigned int *cols);
void writeImage(char *filePath, float *grayscale, unsigned int rows, unsigned int cols);

int main()
{
    float *grayscale = 0;
    unsigned int rows, cols;

    grayscale = readLikeGrayScale("lena.bmp", &rows, &cols);
    writeImage("afterRead.bmp", grayscale, rows, cols);
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0,
                              cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, cols, rows));

    checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, grayscale, rows * cols * sizeof(float),
                                      cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;

    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

    float *dev_output, *output;
    output = (float *)calloc(rows * cols, sizeof(float));
    cudaMalloc(&dev_output, rows * cols * sizeof(float));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
                 (rows + dimBlock.y - 1) / dimBlock.y);
    saltAndPepperWithCuda<<<dimGrid, dimBlock>>>(dev_output, cols, rows);
    checkCudaErrors(cudaMemcpy(output, dev_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    writeImage("result.bmp", output, rows, cols);
    cudaFreeArray(cuArray);
    cudaFree(dev_output);
    return 0;
}

float *readLikeGrayScale(char *filePathInput, unsigned int *rows, unsigned int *cols)
{
    BMP Input;
    Input.ReadFromFile(filePathInput);
    *rows = Input.TellHeight();
    *cols = Input.TellWidth();
    float *grayscale = (float *)calloc(*rows * *cols, sizeof(float));
    for (int j = 0; j < *rows; j++)
    {
        for (int i = 0; i < *cols; i++)
        {
            float gray = (float)floor(0.299 * Input(i, j)->Red +
                                      0.587 * Input(i, j)->Green +
                                      0.114 * Input(i, j)->Blue);
            grayscale[j * *cols + i] = gray;
        }
    }
    return grayscale;
}

void writeImage(char *filePath, float *grayscale, unsigned int rows, unsigned int cols)
{
    BMP Output;
    Output.SetSize(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            RGBApixel pixel;
            pixel.Red = grayscale[i * cols + j];
            pixel.Green = grayscale[i * cols + j];
            pixel.Blue = grayscale[i * cols + j];
            pixel.Alpha = 0;
            Output.SetPixel(j, i, pixel);
        }
    }
    Output.WriteToFile(filePath);
}