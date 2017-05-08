#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "matrix.h"

double randomNormal(double mu, double sigma) {

    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return (mu + sigma * (double) X2);
    }

    do {

        U1 = -1 + ((double) rand() / RAND_MAX) * 2;
        U2 = -1 + ((double) rand() / RAND_MAX) * 2;
        W = pow(U1, 2) + pow(U2, 2);
    } while (W >=1 || W == 0);

    mult = sqrt((-2 * log(W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    call = !call;
    return (mu + sigma * (double) X1);
} 

matrix * randnMatrix(int rows, int cols, double mu, double sigma) {
    if (rows <= 0 || cols <= 0) return NULL;

    matrix *m = (matrix*)malloc(sizeof(matrix));

    m->rows = rows;
    m->cols = cols;

    m->data = (double*)malloc(rows * cols * sizeof(double));

    int i;

    for (i = 0; i < rows*cols; i++){
        m->data[i] = randomNormal(mu, sigma);
    }

    return m;
}

matrix * newMatrix(int rows, int cols) {

    if (rows <= 0 || cols <= 0) return NULL;

    matrix *m = (matrix*)malloc(sizeof(matrix));

    m->rows = rows;
    m->cols = cols;

    m->data = (double*)malloc(rows * cols * sizeof(double));

    int i;

    for (i = 0; i < rows*cols; i++){
        m->data[i] = 0.0;
    }

    return m;
}

int deleteMatrix(matrix * mtx) {
    if (mtx == NULL) return -1;

    assert (mtx->data);
    free(mtx->data);
    free(mtx);
    return 0;
}

// 按行存储矩阵
#define ELEM(mtx, row, col) \
    mtx->data[(row-1) * mtx->cols + (col-1)]

matrix * copyMatrix(matrix * mtx) {
    if (mtx == NULL) return NULL;

    matrix * cp = newMatrix(mtx->rows, mtx->cols);

    memcpy(cp->data, mtx->data, mtx->rows * mtx->cols * sizeof(double));

    return cp;
}

int setElement(matrix * mtx, int row, int col, double val) {
    if (mtx == NULL) return -1;
    assert(mtx->data);
    if (row <= 0 || row > mtx->rows || col <= 0 || col > mtx->cols) return -2;

    ELEM(mtx, row, col) = val;

    return 0;
}

double getElement(matrix * mtx, int row, int col) {
    if (mtx == NULL) return -1;
    assert (mtx->data);
    if (row <= 0 || row > mtx->rows || col <=0 || col > mtx->cols) return -2;

    return ELEM(mtx, row, col);
}

int nRows(matrix * mtx) {
    if (mtx == NULL) return -1;
    return mtx->rows;
}

int nCols(matrix * mtx) {
    if (mtx == NULL) return -1;
    return mtx->cols;
}

int printMatrix(matrix * mtx) {
    if (mtx == NULL) return -1;

    int row, col;
    for (row = 1; row <= mtx->rows; row++) {
        for (col = 1; col <= mtx->cols; col++) {
            printf("% G ", ELEM(mtx, row, col));
        }
        printf("\n");
    }
}

int transpose(matrix * in, matrix * out) {
    if (in == NULL || out == NULL) return -1;
    if (in->rows != out->cols || in->cols != out->rows) return -2;

    int row, col;
    for (row = 1; row <= in->rows; row++)
        for (col = 1; col <= in->cols; col++)
            ELEM(out, col, row) = ELEM(in, row, col);
    return 0;
}

matrix *transpose1(matrix * in) {
    if (in == NULL) return NULL;

    matrix *out = newMatrix(in->cols, in->rows);
    int row, col;
    for (row = 1; row <= in->rows; row++)
        for (col = 1; col <= in->cols; col++)
            ELEM(out, col, row) = ELEM(in, row, col);
    return out;
}

int swap(double *e1, double *e2) {
    if (e1 == NULL || e2 == NULL) return -1;
    double temp;
    temp = *e1;
    *e1 = *e2;
    *e2 = temp;
    return 0;
}

int transposeSelf(matrix * mtx) {
    if (mtx == NULL) return -1;

    int old_rows, old_cols, index, old_index;
    old_rows = mtx->rows;
    old_cols = mtx->cols;
    //交换行列数
    mtx->rows = old_cols;
    mtx->cols = old_rows;
    int row, col, r, c;
    for (row = 1; row <= mtx->rows; row++) {
        for (col = 1; col <= mtx->cols; col++) {
            // 当前矩阵（即转置后的矩阵）元素位置row,col,step = new_cols
            index = mtx->cols * (row - 1) + col;
            // 当前元素对应的原矩阵中元素的初始位置col,row,step = mtx->cols
            old_index = old_cols * (col - 1) + row;
            //printf("index: %d\told_index: %d\n", index, old_index);
            // 寻找原始矩阵对应元素的实际位置
            while (index > old_index) {
                r = old_index % mtx->cols == 0 ?
                    (old_index / mtx->cols) :
                    (old_index / mtx->cols + 1);
                c = old_index - mtx->cols * (r - 1);
                old_index = old_cols * (c - 1) + r;
                //printf("index: %d\told_index: %d\n", index, old_index);
            }
            // 恰好在应该放置的位置，不用交换
            if (index == old_index) continue;

            swap(&(mtx->data[index-1]), &(mtx->data[old_index-1]));
            //swap(&(ELEM(mtx, row, col)), &(ELEM(mtx, c ,r))); // ERROR: 不能对宏取地址
            //printf("index: %d\n", index);
            //printMatrix(mtx);
        }
    }
    return 0;
}

int sum(matrix * mtx1, matrix * mtx2, matrix * sum) {
    if (mtx1 == NULL || mtx2 == NULL || sum == NULL) return -1;
    if (mtx1->rows != mtx2->rows ||
        mtx1->rows != sum->rows ||
        mtx1->cols != mtx2->cols ||
        mtx1->cols != sum->cols)
        return -2;

    int row, col;
    for (row = 1; row <= mtx1->rows; row++)
        for (col = 1; col <= mtx1->cols; col++)
            ELEM(sum, row, col) = ELEM(mtx1, row, col) + ELEM(mtx2, row, col);

    return 0;
}

matrix *sum1(matrix * mtx1, matrix * mtx2) {
    if (mtx1 == NULL || mtx2 == NULL) return NULL;
    if (mtx1->rows != mtx2->rows || mtx1->cols != mtx2->cols) return NULL;

    matrix *sum = newMatrix(mtx1->rows, mtx1->cols);
    int row, col;
    for (row = 1; row <= mtx1->rows; row++)
        for (col = 1; col <= mtx1->cols; col++)
            ELEM(sum, row, col) = ELEM(mtx1, row, col) + ELEM(mtx2, row, col);

    return sum;
}

int minus(matrix *m1, matrix *m2, matrix *result) {
    if (m1 == NULL || m2 == NULL || result == NULL) return -1;
    if (m1->rows != m2->rows ||
        m1->rows != result->rows ||
        m1->cols != m2->cols ||
        m1->cols != result->cols)
        return -2;

    int row, col;
    for (row = 1; row <= result->rows; row++)
        for (col = 1; col <= result->cols; col++)
            ELEM(result, row, col) = ELEM(m1, row, col) - ELEM(m2, row, col);
    return 0;
}

matrix *minus1(matrix *m1, matrix *m2) {
    if (m1 == NULL || m2 == NULL) return NULL;
    if (m1->rows != m2->rows || m1->cols != m2->cols) return NULL;

    matrix * result = newMatrix(m1->rows, m1->cols);
    int row, col;
    for (row = 1; row <= result->rows; row++)
        for (col = 1; col <= result->cols; col++)
            ELEM(result, row, col) = ELEM(m1, row, col) - ELEM(m2, row, col);
    return result;
}


int sumSelf(matrix * mtx, double *sum) {
    if (mtx == NULL || sum == NULL) return -1;
    *sum = 0.0;
    int row, col;
    for (row = 1; row <= mtx->rows; row++)
        for (col = 1; col <= mtx->cols; col++)
            *sum = *sum + ELEM(mtx, row, col);
    return 0;
}

double sumSelf1(matrix * mtx) {
    if (mtx == NULL) return -1;
    double sum = 0.0;
    int row, col;
    for (row = 1; row <= mtx->rows; row++)
        for (col = 1; col <= mtx->cols; col++)
            sum += ELEM(mtx, row, col);
    return sum;
}

int product(matrix * mtx1, matrix * mtx2, matrix * prod) {
    if (mtx1 == NULL || mtx2 == NULL || prod == NULL) return -1;
    if (mtx1->cols != mtx2->rows ||
        mtx1->rows != prod->rows ||
        mtx2->cols != prod->cols)
        return -2;
    
    int row, col, k;
    for (row = 1; row <= mtx1->rows; row++) {
        for (col = 1; col <= mtx2->cols; col++) {
            double val = 0.0;
            for (k = 1; k <= mtx2->rows; k++) {
                val += ELEM(mtx1, row, k) * ELEM(mtx2, k, col);
            }
            ELEM(prod, row, col) = val;
        }
    }
    return 0;
}

matrix *product1(matrix * mtx1, matrix * mtx2) {
    if (mtx1 == NULL || mtx2 == NULL) return NULL;
    if (mtx1->cols != mtx2->rows) return NULL;
    
    matrix *prod = newMatrix(mtx1->rows, mtx2->cols);
    int row, col, k;
    for (row = 1; row <= prod->rows; row++) {
        for (col = 1; col <= prod->cols; col++) {
            double val = 0.0;
            for (k = 1; k <= mtx2->rows; k++) {
                val += ELEM(mtx1, row, k) * ELEM(mtx2, k, col);
            }
            ELEM(prod, row, col) = val;
        }
    }
    return prod;
}

int dotProduct(matrix *v1, matrix *v2, double * prod) {
    if (v1 == NULL || v2 == NULL || prod == NULL) return -1;
    if (v1->cols != 1 || v2->cols != 1) return -2;
    if (v1->rows != v2->rows) return -3;

    *prod = 0.0;
    int i;
    for (i = 1; i <= v1->rows; i++)
        *prod += ELEM(v1, i, 1) * ELEM(v2, i, 1);
    return 0;
}

int scalarProduct(matrix *m1, matrix *m2, matrix *m) {
    if (m1 == NULL || m2 == NULL || m == NULL) return -1;
    if (m1->rows != m2->rows ||
        m1->rows != m->rows ||
        m1->cols != m2->cols ||
        m1->cols != m->cols)
        return -2;
    int row, col;
    for (row = 1; row <= m->rows; row++)
        for (col = 1; col <= m->cols; col++)
            ELEM(m, row, col) = ELEM(m1, row, col) * ELEM(m2, row, col);

    return 0;
}

matrix *scalarProduct1(matrix *m1, matrix *m2) {
    if (m1 == NULL || m2 == NULL) return NULL;
    if (m1->rows != m2->rows || m1->cols != m2->cols) return NULL;
    matrix *m = newMatrix(m1->rows, m1->cols);
    int row, col;
    for (row = 1; row <= m->rows; row++)
        for (col = 1; col <= m->cols; col++)
            ELEM(m, row, col) = ELEM(m1, row, col) * ELEM(m2, row, col);

    return m;
}

int identity(matrix * mtx) {
    if (mtx == NULL || mtx->rows != mtx->cols) return -1;
    int row, col;
    for (row = 1; row <= mtx->rows; row++)
        for (col = 1; col <= mtx->cols; col++)
            if (row == col)
                ELEM(mtx, row, col) = 1.0;
            else
                ELEM(mtx, row, col) = .0;
    return 0;
}


int funcMatrix(matrix *mtx, double (*func)(double x)) {
    if (mtx == NULL) return -1;

    int row, col;
    double elem;
    for (row = 1; row <= mtx->rows; row++) {
        for (col = 1; col <= mtx->cols; col++) {
            elem = func(getElement(mtx, row, col));
            setElement(mtx, row, col, elem);
        }

    }
}

matrix *funcMatrix1(matrix *mtx, double (*func)(double x)) {
    if (mtx == NULL) return NULL;

    matrix *new = copyMatrix(mtx);
    int row, col;
    double elem;
    for (row = 1; row <= new->rows; row++) {
        for (col = 1; col <= new->cols; col++) {
            elem = func(getElement(mtx, row, col));
            setElement(new, row, col, elem);
        }
    }
    return new;
}

int multiplyMatrix(matrix *mtx, double factor) {
    if (mtx == NULL) return -1;
    int row, col;
    double elem;
    for (row = 1; row <= mtx->rows; row++) {
        for (col = 1; col <= mtx->cols; col++) {
            elem = getElement(mtx, row, col);
            elem *= factor;
            setElement(mtx, row, col, elem);
        }
    }
    return 0;
}
// 返回一个向量的最大元素下标
int maxVector(matrix *vector) {
    if (vector == NULL || vector->cols != 1) return -1;

    int row, max_index;
    double elem, max;
    max =  -INFINITY;
    for (row = 1; row <= vector->rows; row++) {
        elem = getElement(vector, row, 1);
        if (elem > max) {
            max = elem;
            max_index = row;
        }
    }
    return max_index;
}

// 返回一个向量的最大元素下标
int minVector(matrix *vector) {
    if (vector == NULL || vector->cols != 1) return -1;

    int row, min_index;
    double elem, min;
    min =  INFINITY;
    for (row = 1; row <= vector->rows; row++) {
        elem = getElement(vector, row, 1);
        if (elem < min) {
            min = elem;
            min_index = row;
        }
    }
    return min_index;
}

