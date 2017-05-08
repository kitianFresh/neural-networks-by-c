#include <stdio.h>
#include <math.h>
#include "matrix.h"

int main() {
    matrix *A, *Ac, *at, *B, *C, *ct, *alpha, *beta, *s, *p, *I, *M, *mis, *v;
    double dp, summ;
    A = newMatrix(3,4);
    int i,j;
    for (i = 1; i <= 3; i++)
        for (j = 1; j<=4; j++)
            setElement(A, i, j, 4 * (i - 1) + j);
    printf("Matrix A:\n");
    printMatrix(A);

    Ac = copyMatrix(A);
    printf("\nCopy of A:\n");
    printMatrix(Ac);

    B = newMatrix(4,2);
    for (i = 1; i <= 4; i++)
        for (j = 1; j <=2; j++)
            setElement(B, i, j, 1);
    printf("\nMatrix B:\n");
    printMatrix(B);

    C = newMatrix(3,2);
    product(A, B, C);
    printf("\nC = A product B:\n");
    printMatrix(C);

    ct = newMatrix(2,3);
    transpose(C, ct);
    printf("\nct transpose of C:\n");
    printMatrix(ct);

    transposeSelf(A);
    printf("\ntransposeSelf of A\n");
    printMatrix(A);

    
    alpha = newMatrix(3,1);
    setElement(alpha, 1, 1, 1.0);
    setElement(alpha, 2, 1, 2.0);
    setElement(alpha, 3, 1, 3.0);
    printf("\nVector alpha:\n");
    printMatrix(alpha);

    beta = newMatrix(3,1);
    setElement(beta, 1, 1, 1.0);
    setElement(beta, 2, 1, 1.0);
    setElement(beta, 3, 1, 1.0);
    printf("\nVector beta:\n");
    printMatrix(beta);

    s = newMatrix(3,1);
    sum(alpha, beta, s);
    printf("\ns = alpha + beta:\n");
    printMatrix(s);



    dotProduct(alpha, beta, &dp);
    printf("\ndp = alpha dotProduct beta:\n");
    printf("% 6.2f \n", dp);
    

    p = newMatrix(4,1);
    product(A, beta, p);
    printf("\np = A dot beta:\n");
    printMatrix(p);


    sumSelf(alpha, &summ);
    printf("\nsumSelf of alpha:\n");
    printf("% 6.2f \n", summ);

    sumSelf(A, &summ);
    printf("\nsumSelf of A:\n");
    printf("% 6.2f \n", summ);


    I = newMatrix(3,3);
    identity(I);
    printf("\nIdentity:\n");
    printMatrix(I);


    //double (*f)(double x);
    //f = exp;
    // 使用math.h，用gcc编译时需要参数 -lm加到最后面;gcc test_matrix.c matrix.c -o test_matrix -lm;
    funcMatrix(I, exp);
    printf("\nexp(I) funcMatrix by exp:\n");
    printMatrix(I);

    M = newMatrix(3,3);
    for (i = 1; i <= 3; i++)
        for (j = 1; j <= 3; j++)
            setElement(M, i, j, 2.);
    printf("\nM:\n");
    printMatrix(M);

    mis = newMatrix(3,3);
    scalarProduct(M, I, mis);
    printf("\nmis = M scalarProduct I:\n");
    printMatrix(mis);

    minus(mis, M, mis);
    printf("\nmis = mis -M:\n");
    printMatrix(mis);

    printf("\n----------------------------------------------------------\n");
    matrix *M1, *M2, *M3, *M4;
    M1 = newMatrix(3,3);
    for (i = 1; i <= 3; i++)
        for (j = 1; j <= 3; j ++)
            setElement(M1, i, j, 2);
    M2 = newMatrix(3,3);
    for (i = 1; i <= 3; i++)
        for (j = 1; j <= 3; j ++)
            setElement(M2, i, j, 1);

    printf("\nM1:\n");
    printMatrix(M1);
    printf("\nM2:\n");
    printMatrix(M2);
    printf("\nM1 -M2:\n");
    printMatrix(minus1(M1, M2));
    printf("\nM1 + M2:\n");
    printMatrix(sum1(M1, M2));
    printf("\nM1 product M2:\n");
    printMatrix(product1(M1, M2));
    printf("\nM1 scalarProduct M2:\n");
    printMatrix(scalarProduct1(M1, M2));
    printf("\nsumSelf M1:\n");
    printf("% 6.2f\n", sumSelf1(M1));

    M3 = newMatrix(3,4);
    for (i = 1; i <= 3; i++)
        for (j = 1; j<=4; j++)
            setElement(M3, i, j, 4 * (i - 1) + j);
    printf("M3:\n");
    printMatrix(M3);

    M4 = newMatrix(4,2);
    for (i = 1; i <= 4; i++)
        for (j = 1; j <=2; j++)
            setElement(M4, i, j, 1);
    printf("\nM4:\n");
    printMatrix(M4);

    printf("\nM3 product M4:\n");
    printMatrix(product1(M3, M4));

    printf("\ntranspose M3:\n");
    printMatrix(transpose1(M3));


    printf("\nmultiplyMatrix M1 by 2.0:\n");
    multiplyMatrix(M1, 2.0);
    printMatrix(M1);

    v = newMatrix(10,1);
    setElement(v, 1, 1, 8);
    setElement(v, 2, 1, 4);
    setElement(v, 3, 1, 6);
    setElement(v, 4, 1, 9);
    setElement(v, 5, 1, 1);
    setElement(v, 6, 1, 7);
    setElement(v, 7, 1, 5);
    setElement(v, 8, 1, 0);
    setElement(v, 9, 1, 3);
    setElement(v, 10, 1, 2);
    printf("\nVector v:\n");
    printMatrix(v);

    printf("\nmaxVector:\n");
    printf("%d\n", maxVector(v));

    printf("\nminVector:\n");
    printf("%d\n", minVector(v));

    transposeSelf(v);
    printMatrix(v);

    matrix * std_normal = randnMatrix(4, 4, 100000, 10);
    printf("\nstandard normal distribution matrix:\n");
    printMatrix(std_normal);

    return 0;

}
