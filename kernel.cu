#include <iostream>

#define WINDOW_SIZE 3
#define WINDOW_LENGHT WINDOW_SIZE *WINDOW_SIZE

texture<float, cudaTextureType2D, cudaReadModeElementType> tex;

__global__ void saltAndPepperWithCuda(float *output, int imageWidth, int imageHeight)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (!(row < imageHeight && col < imageWidth)) {
        return;
    }

    float filter[WINDOW_LENGHT] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int x = 0; x < WINDOW_SIZE; x++)
    {
        for (int y = 0; y < WINDOW_SIZE; y++)
        {
            filter[x * WINDOW_SIZE + y] = tex2D(tex, col + y - 1, row + x - 1);
        }
    }
    for (int i = 0; i < WINDOW_LENGHT; i++)
    {
        for (int j = i + 1; j < WINDOW_LENGHT; j++)
        {
            if (filter[i] > filter[j])
            {
                float tmp = filter[i];
                filter[i] = filter[j];
                filter[j] = tmp;
            }
        }
    }
    output[row * imageWidth + col] = filter[(int)(WINDOW_LENGHT / 2)];
}

#define kernel saltAndPepperWithCuda
#include "main.h"