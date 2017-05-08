#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "matrix.h"
#include <stdbool.h>

int generate_data(matrix * datas[], matrix * labels[], int size) {
    int a, b;
    for (int i=0; i<size; i++) {
        a = (double) rand() / RAND_MAX * 10;
        b = (double) rand() / RAND_MAX * 10;
        matrix * v = newMatrix(2, 1);
        setElement(v, 1, 1, a);
        setElement(v, 2, 1, b);
        datas[i] = v;
        matrix * l = newMatrix(1,1);
        setElement(l, 1, 1, a+b);
        labels[i] = l;
    }
}

int main() {

    int train_size = 50000;
    int test_size = 2000;
    matrix * train_datas[train_size];
    matrix * train_labels[train_size];
    matrix * test_datas[test_size];
    matrix * test_labels[test_size];

    generate_data(train_datas, train_labels, train_size);
    generate_data(test_datas, test_labels, test_size);

    int layers = 3;
    int sizes[3] = {2, 10, 1};
    init(sizes, layers);
    SGD(train_datas, train_labels, train_size, 30, 10, 0.1, test_datas, test_labels, test_size, true);

    matrix * datas[100];
    matrix * y_hat[100];
    int n,i;
    scanf("%d", &n);
    i = 0;
    while (i < n) {
        double a,b;
        scanf("%lf %lf", &a, &b);
        matrix * v = newMatrix(2,1);
        setElement(v, 1, 1, a);
        setElement(v, 2, 1, b);
        datas[i] = v;
    }
    predict(datas, 1, y_hat);
    printMatrix(y_hat);
    return 0;
}