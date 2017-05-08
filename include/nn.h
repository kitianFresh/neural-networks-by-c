#ifndef __NN_H__
#define __NN_H__
#include "matrix.h"
#include <stdbool.h>

int contains(int *array, int size, int elem);

int randomShuffle(int *array, int size);

int init(int *sizeArray, int n_layers);

double sigmoid(double x);

double sigmoid_prime(double x);

matrix * cost_derivative(matrix *output_activations, matrix *y);

matrix *feedforward(matrix *in);

int evaluate(matrix *test_images[], matrix *test_labels[], int test_size);

int update_mini_batch(matrix *mini_batch_images[], matrix *mini_batch_labels[], int mini_batch_size, double eta);

int backprop(matrix *x, matrix *y, matrix *nabla_w[], matrix *nabla_b[]);

int SGD(matrix *train_images[], matrix *train_labels[], int train_size, int epochs, int mini_batch_size, double eta,
		matrix *test_images[], matrix *test_labels[], int test_size, bool regression);



#endif