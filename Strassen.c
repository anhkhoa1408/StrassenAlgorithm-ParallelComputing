int **strassen(int **matrixA, int **matrixB, int **matrixC, int size)
{

    if (size == 1)
    {
        matrixC[0][0] = matrixA[0][0] * matrixB[0][0];
        return matrixC;
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
        subtractMatrix(b12, b22, matrixAResult, newSize);   // b12 - b22
        strassen(a11, matrixAResult, p1, newSize);          // p1 = a11 * (b12 - b22)

        sumMatrix(a11, a12, matrixAResult, newSize);        // a11 + a12
        strassen(matrixAResult, b22, p2, newSize);          // p2 = (a11 + a12) * b22

        sumMatrix(a21, a22, matrixAResult, newSize);        // a21 + a22
        strassen(matrixAResult, b11, p3, newSize);          // p3 = (a21 + a22) * b11

        subtractMatrix(b21, b11, matrixBResult, newSize);   // b21 - b11
        strassen(a22, matrixBResult, p4, newSize);          // p4 = (a22) * (b21 - b11)

        sumMatrix(a11, a22, matrixAResult, newSize);         // a11 + a22
        sumMatrix(b11, b22, matrixBResult, newSize);         // b11 + b22
        strassen(matrixAResult, matrixBResult, p5, newSize); // p5 = (a11 + a22) * (b11 + b22)

        subtractMatrix(a12, a22, matrixAResult, newSize);    // a12 - a22
        sumMatrix(b21, b22, matrixBResult, newSize);         // b21 + b22
        strassen(matrixAResult, matrixBResult, p6, newSize); // p6 = (a12 - a22) * (b21 + b22)

        subtractMatrix(a11, a21, matrixAResult, newSize);    // a11 - a21
        sumMatrix(b11, b12, matrixBResult, newSize);         // b11 + b12
        strassen(matrixAResult, matrixBResult, p7, newSize); // p7 = (a11 - a21) * (b11 + b12)

        // calculating c11, c12, c21, c22:

        //c11
        sumMatrix(p5, p4, matrixAResult, newSize);              // p5 + p4
        sumMatrix(matrixAResult, p6, matrixBResult, newSize);   // p5 + p4 + p6
        subtractMatrix(matrixBResult, p2, c11, newSize);        // c11 = p5 + p4 + p6 - p2

        //c12
        sumMatrix(p1, p2, c12, newSize);                        // c12 = p1 + p2

        //c21
        sumMatrix(p3, p4, c21, newSize);                        // c21 = p3 + p4

        //c22
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