#ifndef __MNIST_READER_H__
#define __MNIST_READER_H__
#include "matrix.h"

char **strsplit(const char* str, const char* delim, size_t* numtokens);

//　给label用
matrix * vectorize(int i, int vectorlen);

int readImageDataAsVectorArray(char *filename, matrix *images[], int num, int vectorlen);

int readLabelDataAsVectorArray(char *filename, matrix *labels[], int num, int vectorlen);

#endif