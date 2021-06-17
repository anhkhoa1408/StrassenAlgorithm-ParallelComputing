#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define LEAF_SIZE 16

bool checkValidNumber(int number) 
{
    return log2(number) == (int)(log2(number)) ? true : false;
}

void randomMatrix(int **matrixA, int **matrixB, int **matrixC, int size)
{
    int i, j;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            matrixA[i][j] = (rand() % (2 * 100 + 1)) - 100;
        }
    }

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            matrixB[i][j] = (rand() % (2 * 100 + 1)) - 100;
        }
    }

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            matrixC[i][j] = 0;
        }
    }
}

int **allocateMatrix(int size)
{
    int **Matrix;
    int i;
    Matrix = (int **)malloc(size * sizeof(int *));
    for (i = 0; i < size; i++)
    {
        Matrix[i] = (int *)malloc(size * sizeof(int));
    }
    return Matrix;
}

int **freeMatrix(int **Matrix, int size)
{
    int i;
    if (Matrix == NULL)
    {
        return (NULL);
    }

    for (i = 0; i < size; i++)
    {
        free(Matrix[i]);
    }
    free(Matrix);
    Matrix = NULL;
    return (NULL);
}

void sumMatrix(int **matrixA, int **matrixB, int **result, int size)
{

    int i, j;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            result[i][j] = matrixA[i][j] + matrixB[i][j];
        }
    }
}

void subtractMatrix(int **matrixA, int **matrixB, int **result, int size)
{

    int i, j;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            result[i][j] = matrixA[i][j] - matrixB[i][j];
        }
    }
}

int **strassenOpenMP(int **matrixA, int **matrixB, int **matrixC, int size, int core)
{
    int i, j, k, nthreads, chunk;
    if (size == LEAF_SIZE)
    {
        for (i = 0; i < LEAF_SIZE; i++)
        {
            for (j = 0; j < LEAF_SIZE; j++)
            {
                for (k = 0; k < LEAF_SIZE; k++)
                {
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
    }
    else
    {
        int newSize = size / 2;
        int **a11, **a12, **a21, **a22;
        int **b11, **b12, **b21, **b22;
        int **c11, **c12, **c21, **c22;
        int **p1, **p2, **p3, **p4, **p5, **p6, **p7;
        // memory allocation:
        a11 = allocateMatrix(newSize);
        a12 = allocateMatrix(newSize);
        a21 = allocateMatrix(newSize);
        a22 = allocateMatrix(newSize);

        b11 = allocateMatrix(newSize);
        b12 = allocateMatrix(newSize);
        b21 = allocateMatrix(newSize);
        b22 = allocateMatrix(newSize);

        c11 = allocateMatrix(newSize);
        c12 = allocateMatrix(newSize);
        c21 = allocateMatrix(newSize);
        c22 = allocateMatrix(newSize);

        p1 = allocateMatrix(newSize);
        p2 = allocateMatrix(newSize);
        p3 = allocateMatrix(newSize);
        p4 = allocateMatrix(newSize);
        p5 = allocateMatrix(newSize);
        p6 = allocateMatrix(newSize);
        p7 = allocateMatrix(newSize);

        int **matrixAResult = allocateMatrix(newSize);
        int **matrixBResult = allocateMatrix(newSize);

        #pragma omp parallel shared(matrixA, matrixB, a11, a12, a21, a22, b11, b12, b21, b22) private(j, i)
        {
            nthreads = core;
            chunk = newSize / nthreads;
            #pragma omp for schedule(guided, chunk)
            //divide the matrix into 4 sub-matrices
            for (i = 0; i < newSize; i++)
            {
                for (j = 0; j < newSize; j++)
                {
                    a11[i][j] = matrixA[i][j];
                    a12[i][j] = matrixA[i][j + newSize];
                    a21[i][j] = matrixA[i + newSize][j];
                    a22[i][j] = matrixA[i + newSize][j + newSize];

                    b11[i][j] = matrixB[i][j];
                    b12[i][j] = matrixB[i][j + newSize];
                    b21[i][j] = matrixB[i + newSize][j];
                    b22[i][j] = matrixB[i + newSize][j + newSize];
                }
            }
        }

        // Calculating p1 to p7:

        #pragma omp parallel sections
        {

            #pragma omp section
            {
                subtractMatrix(b12, b22, matrixAResult, newSize); // b12 - b22
                strassenOpenMP(a11, matrixAResult, p1, newSize, core);  // p1 = a11 * (b12 - b22)
            }

            #pragma omp section
            {
                sumMatrix(a11, a12, matrixAResult, newSize);     // a11 + a12
                strassenOpenMP(matrixAResult, b22, p2, newSize, core); // p2 = (a11 + a12) * b22
            }

            #pragma omp section
            {
                sumMatrix(a21, a22, matrixAResult, newSize);     // a21 + a22
                strassenOpenMP(matrixAResult, b11, p3, newSize, core); // p3 = (a21 + a22) * b11
            }

            #pragma omp section
            {
                subtractMatrix(b21, b11, matrixBResult, newSize); // b21 - b11
                strassenOpenMP(a22, matrixBResult, p4, newSize, core);  // p4 = (a22) * (b21 - b11)
            }

            #pragma omp section
            {
                sumMatrix(a11, a22, matrixAResult, newSize);               // a11 + a22
                sumMatrix(b11, b22, matrixBResult, newSize);               // b11 + b22
                strassenOpenMP(matrixAResult, matrixBResult, p5, newSize, core); // p5 = (a11 + a22) * (b11 + b22)
            }

            #pragma omp section
            {
                subtractMatrix(a12, a22, matrixAResult, newSize);          // a12 - a22
                sumMatrix(b21, b22, matrixBResult, newSize);               // b21 + b22
                strassenOpenMP(matrixAResult, matrixBResult, p6, newSize, core); // p6 = (a12 - a22) * (b21 + b22)
            }

            #pragma omp section
            {
                subtractMatrix(a11, a21, matrixAResult, newSize);          // a11 - a21
                sumMatrix(b11, b12, matrixBResult, newSize);               // b11 + b12
                strassenOpenMP(matrixAResult, matrixBResult, p7, newSize, core); // p7 = (a11 - a21) * (b11 + b12)
            }
        }

        // calculating c11, c12, c21 and c22:

        //c11
        sumMatrix(p5, p4, matrixAResult, newSize);            // p5 + p4
        sumMatrix(matrixAResult, p6, matrixBResult, newSize); // p5 + p4 + p6
        subtractMatrix(matrixBResult, p2, c11, newSize);      // c11 = p5 + p4 + p6 - p2

        //c12
        sumMatrix(p1, p2, c12, newSize); // c12 = p1 + p2

        //c21
        sumMatrix(p3, p4, c21, newSize); // c21 = p3 + p4

        //c22
        sumMatrix(p1, p5, matrixAResult, newSize);                 // p1 + p5
        subtractMatrix(matrixAResult, p3, matrixBResult, newSize); // p1 + p5 - p3
        subtractMatrix(matrixBResult, p7, c22, newSize);           // c22 = p1 + p5 - p3 - p7

        //combine the result into one matrix
        #pragma omp parallel shared(matrixC, c11, c12, c21, c22) private(j, i)
        {
            for (i = 0; i < newSize; i++)
            {
                for (j = 0; j < newSize; j++)
                {
                    matrixC[i][j] = c11[i][j];
                    matrixC[i][j + newSize] = c12[i][j];
                    matrixC[i + newSize][j] = c21[i][j];
                    matrixC[i + newSize][j + newSize] = c22[i][j];
                }
            }
        }

        // deallocating memory
        a11 = freeMatrix(a11, newSize);
        a12 = freeMatrix(a12, newSize);
        a21 = freeMatrix(a21, newSize);
        a22 = freeMatrix(a22, newSize);

        b11 = freeMatrix(b11, newSize);
        b12 = freeMatrix(b12, newSize);
        b21 = freeMatrix(b21, newSize);
        b22 = freeMatrix(b22, newSize);

        c11 = freeMatrix(c11, newSize);
        c12 = freeMatrix(c12, newSize);
        c21 = freeMatrix(c21, newSize);
        c22 = freeMatrix(c22, newSize);

        p1 = freeMatrix(p1, newSize);
        p2 = freeMatrix(p2, newSize);
        p3 = freeMatrix(p3, newSize);
        p4 = freeMatrix(p4, newSize);
        p5 = freeMatrix(p5, newSize);
        p6 = freeMatrix(p6, newSize);
        p7 = freeMatrix(p7, newSize);

        matrixAResult = freeMatrix(matrixAResult, newSize);
        matrixBResult = freeMatrix(matrixBResult, newSize);
    }
    return matrixC;
}

int **strassen(int **matrixA, int **matrixB, int **matrixC, int size)
{
    if (size == LEAF_SIZE)
    {
        int i, j, k;
        for (i = 0; i < LEAF_SIZE; i++)
        {
            for (j = 0; j < LEAF_SIZE; j++)
            {
                for (k = 0; k < LEAF_SIZE; k++)
                {
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
    }
    else
    {
        int newSize = size / 2;
        int **a11, **a12, **a21, **a22;
        int **b11, **b12, **b21, **b22;
        int **c11, **c12, **c21, **c22;
        int **p1, **p2, **p3, **p4, **p5, **p6, **p7;

        // memory allocation:
        a11 = allocateMatrix(newSize);
        a12 = allocateMatrix(newSize);
        a21 = allocateMatrix(newSize);
        a22 = allocateMatrix(newSize);

        b11 = allocateMatrix(newSize);
        b12 = allocateMatrix(newSize);
        b21 = allocateMatrix(newSize);
        b22 = allocateMatrix(newSize);

        c11 = allocateMatrix(newSize);
        c12 = allocateMatrix(newSize);
        c21 = allocateMatrix(newSize);
        c22 = allocateMatrix(newSize);

        p1 = allocateMatrix(newSize);
        p2 = allocateMatrix(newSize);
        p3 = allocateMatrix(newSize);
        p4 = allocateMatrix(newSize);
        p5 = allocateMatrix(newSize);
        p6 = allocateMatrix(newSize);
        p7 = allocateMatrix(newSize);

        int **matrixAResult = allocateMatrix(newSize);
        int **matrixBResult = allocateMatrix(newSize);

        int i, j;

        //dividing the matrices in 4 sub-matrices:
        for (i = 0; i < newSize; i++)
        {
            for (j = 0; j < newSize; j++)
            {
                a11[i][j] = matrixA[i][j];
                a12[i][j] = matrixA[i][j + newSize];
                a21[i][j] = matrixA[i + newSize][j];
                a22[i][j] = matrixA[i + newSize][j + newSize];

                b11[i][j] = matrixB[i][j];
                b12[i][j] = matrixB[i][j + newSize];
                b21[i][j] = matrixB[i + newSize][j];
                b22[i][j] = matrixB[i + newSize][j + newSize];
            }
        }

        // Calculating p1 to p7:
        subtractMatrix(b12, b22, matrixAResult, newSize);    // b12 - b22
        strassen(a11, matrixAResult, p1, newSize);           // p1 = a11 * (b12 - b22)

        sumMatrix(a11, a12, matrixAResult, newSize);         // a11 + a12
        strassen(matrixAResult, b22, p2, newSize);           // p2 = (a11 + a12) * b22

        sumMatrix(a21, a22, matrixAResult, newSize);         // a21 + a22
        strassen(matrixAResult, b11, p3, newSize);           // p3 = (a21 + a22) * b11

        subtractMatrix(b21, b11, matrixBResult, newSize);    // b21 - b11
        strassen(a22, matrixBResult, p4, newSize);           // p4 = (a22) * (b21 - b11)

        sumMatrix(a11, a22, matrixAResult, newSize);         // a11 + a22
        sumMatrix(b11, b22, matrixBResult, newSize);         // b11 + b22
        strassen(matrixAResult, matrixBResult, p5, newSize); // p5 = (a11 + a22) * (b11 + b22)

        subtractMatrix(a12, a22, matrixAResult, newSize);    // a12 - a22
        sumMatrix(b21, b22, matrixBResult, newSize);         // b21 + b22
        strassen(matrixAResult, matrixBResult, p6, newSize); // p6 = (a12 - a22) * (b21 + b22)

        subtractMatrix(a11, a21, matrixAResult, newSize);    // a11 - a21
        sumMatrix(b11, b12, matrixBResult, newSize);         // b11 + b12
        strassen(matrixAResult, matrixBResult, p7, newSize); // p7 = (a11 - a21) * (b11 + b12)

        // Calculating c11, c12, c21, c22:

        // c11
        sumMatrix(p5, p4, matrixAResult, newSize);              // p5 + p4
        sumMatrix(matrixAResult, p6, matrixBResult, newSize);   // p5 + p4 + p6
        subtractMatrix(matrixBResult, p2, c11, newSize);        // c11 = p5 + p4 + p6 - p2

        // c12
        sumMatrix(p1, p2, c12, newSize);                        // c12 = p1 + p2

        // c21
        sumMatrix(p3, p4, c21, newSize);                        // c21 = p3 + p4

        // c22
        sumMatrix(p1, p5, matrixAResult, newSize);                 // p1 + p5
        subtractMatrix(matrixAResult, p3, matrixBResult, newSize); // p1 + p5 - p3
        subtractMatrix(matrixBResult, p7, c22, newSize);           // p1 + p5 - p3 - p7

        // Grouping the results obtained in a single matrix:
        for (i = 0; i < newSize; i++)
        {
            for (j = 0; j < newSize; j++)
            {
                matrixC[i][j] = c11[i][j];
                matrixC[i][j + newSize] = c12[i][j];
                matrixC[i + newSize][j] = c21[i][j];
                matrixC[i + newSize][j + newSize] = c22[i][j];
            }
        }

        // deallocating memory (free):
        a11 = freeMatrix(a11, newSize);
        a12 = freeMatrix(a12, newSize);
        a21 = freeMatrix(a21, newSize);
        a22 = freeMatrix(a22, newSize);

        b11 = freeMatrix(b11, newSize);
        b12 = freeMatrix(b12, newSize);
        b21 = freeMatrix(b21, newSize);
        b22 = freeMatrix(b22, newSize);

        c11 = freeMatrix(c11, newSize);
        c12 = freeMatrix(c12, newSize);
        c21 = freeMatrix(c21, newSize);
        c22 = freeMatrix(c22, newSize);

        p1 = freeMatrix(p1, newSize);
        p2 = freeMatrix(p2, newSize);
        p3 = freeMatrix(p3, newSize);
        p4 = freeMatrix(p4, newSize);
        p5 = freeMatrix(p5, newSize);
        p6 = freeMatrix(p6, newSize);
        p7 = freeMatrix(p7, newSize);

        matrixAResult = freeMatrix(matrixAResult, newSize);
        matrixBResult = freeMatrix(matrixBResult, newSize);
    }
    return matrixC;
}

int **Naive(int **MatrixA, int **MatrixB, int **MatrixC, int size)
{
    int i, j, k;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            for (k = 0; k < size; k++)
            {
                MatrixC[i][j] += MatrixA[i][k] * MatrixB[k][j];
            }
        }
    }
    return MatrixC;
}

int main(int argc, char *argv[])
{
    int tid, nthreads;
    int i, j, k;
    clock_t t;
    srand(time(NULL));
    
    int **matrixA, **matrixB, **matrixC;

    
    int option;
    int sizeOfMatrix;
    printf("Input size of matrix: ");
    scanf("%d", &sizeOfMatrix);

    while (!checkValidNumber(sizeOfMatrix))
    {
        printf("\nPlease input size of matrix which is square of 2\n");
        printf("Input size of matrix: ");
        scanf("%d", &sizeOfMatrix);
    }

    printf("\n1: Naive Algorithm\n");
    printf("2: Sequential Strassen Algorithm\n");
    printf("3: Parallel Strassen Algorithm\n");
    printf("Please choose Matrix Multiplication Algorithm: ");
    scanf("%d", &option);

    if (option == 1)
    {
        matrixA = allocateMatrix(sizeOfMatrix);
        matrixB = allocateMatrix(sizeOfMatrix);
        matrixC = allocateMatrix(sizeOfMatrix);
        randomMatrix(matrixA, matrixB, matrixC, sizeOfMatrix);
        t = clock();
        Naive(matrixA, matrixB, matrixC, sizeOfMatrix);
        t = clock() - t;
        double naive_time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
        printf("\nNaive time taken %.12f\n", naive_time_taken);
    }
    else if (option == 2)
    {
        matrixA = allocateMatrix(sizeOfMatrix);
        matrixB = allocateMatrix(sizeOfMatrix);
        matrixC = allocateMatrix(sizeOfMatrix);
        randomMatrix(matrixA, matrixB, matrixC, sizeOfMatrix);
        t = clock();
        strassen(matrixA, matrixB, matrixC, sizeOfMatrix);
        t = clock() - t;
        double strassen_time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
        printf("\nSequential Strassen time taken %.12f\n", strassen_time_taken);
    }
    else if (option == 3)
    {
        int core;
        printf("Input number of core: ");
        scanf("%d", &core);

        while (!checkValidNumber(core))
        {
            printf("Please input number of core which is square of 2 \n");
            printf("Input number of core: ");
            scanf("%d", &core);
        }

        tid = omp_get_thread_num();

        if (tid == 0)
        {
            matrixA = allocateMatrix(sizeOfMatrix);
            matrixB = allocateMatrix(sizeOfMatrix);
            matrixC = allocateMatrix(sizeOfMatrix);
        }
        randomMatrix(matrixA, matrixB, matrixC, sizeOfMatrix);
        double start_time = omp_get_wtime();
        strassenOpenMP(matrixA, matrixB, matrixC, sizeOfMatrix, core);
        double parallel_time_taken = omp_get_wtime() - start_time;
        printf("\nParallel Strassen time taken %.12f \n", parallel_time_taken);
    }

    freeMatrix(matrixA, sizeOfMatrix);
    freeMatrix(matrixB, sizeOfMatrix);
    freeMatrix(matrixC, sizeOfMatrix);

    return 0;
}