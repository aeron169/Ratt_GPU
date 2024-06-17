#include "exercice.h"

/**
 * L’ensemble de Julia est d´efini par la suite
zn+1 = zn^2 + c (1)
o`u zn et c sont des nombre complexes. Le nombre c est consid´er´e comme constant et vaut ici c = −0.5 + 0.6i. La
suite est initialis´ee `a z0 qui correspond aux coordonn´ees d’un pixel d’une image o`u la partie r´eelle correspond `a
l’axe x et la partie imaginaire `a l’axe y. L’objectif est de g´en´erer une image de N colonnes et M lignes o`u chaque
pixel d’indice (i, j) indique si la suite de l’Equation 1 converge ou pas. On consid`ere que le suite converge d´es lors
que |zn|
2 < 2, ∀n ≤ 100, o`u |zn| est le module de zn. On se concentre sur la zone x ∈ (−1.5, +1.5) en horizontal et
y ∈ (−1.0, +1.0) en vertical. On note C le nombre de canaux dans l’image g´en´er´ee, valant 1 dans les deux premi`eres
´etapes, et 3 (pour les composantes rouge, vert et bleu) dans la troisi`eme et derni`ere ´etape.
Le fichier exercice.h contient la fonction get ptr pour acc´eder au contenu de l’image. Dans tous les exercices,
la m´emoire doit ˆetre allou´ee sur GPU en utilisant la fonction cudaMallocPitch. Les noyaux CUDA 2D sont tous
lanc´es avec BLOCK SIZE×BLOCK SIZE threads par block.
 */

/*
Le premi`ere ´etape consiste `a g´en´erer une image en noir et blanc o`u un pixel est blanc (= 1.0) si la suite converge,
ou noir (= 0.0) sinon.
1. Dans le fichier exercice.cu, compl´eter le noyau CUDA 2D kernel generate1 qui traite un pixel de l’image
binaire g´en´er´ee avec C = 1.
*/
__global__ void kernel_generate1(int N, int M, int C, int pitch, float *img)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M)
    {
        float x = -1.5 + 3.0 * i / N;
        float y = -1.0 + 2.0 * j / M;
        int n = 0;
        while (n < 100 && x * x + y * y < 4.0)
        {
            float x_new = x * x - y * y - 0.5;
            float y_new = 2.0 * x * y + 0.6;
            x = x_new;
            y = y_new;
            n++;
        }
        img[j * pitch + i] = n < 100 ? 1.0 : 0.0;
    }
}

/**
 * Dans le fichier exercice.cu, compl ́eter la fonction generate1 pour allouer de la m ́emoire sur GPU, lancer
le noyau CUDA avec BLOCK SIZE×BLOCK SIZE threads par block, rapatrier les donn ́ees sur CPU, lib ́erer la
m ́emoire allou ́ee sur GPU, puis retourner le r ́esultat
 */
float *generate1(int N, int M, int C)
{
    float *img = nullptr;
    float *d_img = nullptr;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, N * sizeof(float), M);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
    kernel_generate1<<<grid_size, block_size>>>(N, M, C, pitch / sizeof(float), d_img);
    img = new float[N * M];
    cudaMemcpy2D(img, N * sizeof(float), d_img, pitch, N * sizeof(float), M, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    return img;
}

__global__ void kernel_generate2(int N, int M, int C, int pitch, float *img)
{
}

float *generate2(int N, int M, int C)
{
    return nullptr;
}

__global__ void kernel_generate3(int N, int M, int C, int pitch, float *img)
{
}

float *generate3(int N, int M, int C)
{
    return nullptr;
}
