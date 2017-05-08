#ifndef __MATRIX_H__
#define __MATRIX_H__

typedef struct {
    int rows;
    int cols;
    double *data;
} matrix;

matrix * randnMatrix(int rows, int cols, double mu, double sigma);

matrix * newMatrix(int rows, int cols);

int deleteMatrix(matrix * mtx);

// 按行存储矩阵
#define ELEM(mtx, row, col) \
    mtx->data[(row-1) * mtx->cols + (col-1)]

matrix * copyMatrix(matrix * mtx);

int setElement(matrix * mtx, int row, int col, double val);

double getElement(matrix * mtx, int row, int col);

int nRows(matrix * mtx);

int nCols(matrix * mtx);

int printMatrix(matrix * mtx);

int transpose(matrix * in, matrix * out);

int swap(double *e1, double *e2);

int transposeSelf(matrix * mtx);

int sum(matrix * mtx1, matrix * mtx2, matrix * sum);

int minus(matrix *m1, matrix *m2, matrix *result);

int sumSelf(matrix * mtx, double *sum);

int product(matrix * mtx1, matrix * mtx2, matrix * prod);

int dotProduct(matrix *v1, matrix *v2, double * prod);

int scalarProduct(matrix *m1, matrix *m2, matrix *m);

int identity(matrix * mtx);

// 对矩阵的每一个元素进行某种函数运算
int funcMatrix(matrix *mtx, double (*func)(double x));

int multiplyMatrix(matrix *mtx, double factor);

int maxVector(matrix *vector);

int minVector(matrix *vector);


matrix *transpose1(matrix * in);

matrix *sum1(matrix * mtx1, matrix * mtx2);

matrix *minus1(matrix *m1, matrix *m2);

double sumSelf1(matrix * mtx);

matrix *product1(matrix * mtx1, matrix * mtx2);

matrix *scalarProduct1(matrix *m1, matrix *m2);

matrix *funcMatrix1(matrix *mtx, double (*func)(double x));

#endif
