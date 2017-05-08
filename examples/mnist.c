#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "matrix.h"


int main() {
	
	matrix *train_images[60000];
	matrix *train_labels[60000];
	matrix *test_images[10000];
	matrix *test_labels[10000];
    char *filetrainimages = "../data/train_images.txt";
    char *filetrainlabels = "../data/train_labels.txt";
    char *filetestimages  = "../data/test_images.txt";
    char *filetestlabels  = "../data/test_labels.txt";;
    printf("reading data......\n");
    readImageDataAsVectorArray(filetrainimages, train_images, 60000, 28*28);    
    readLabelDataAsVectorArray(filetrainlabels, train_labels, 60000, 10);
    readImageDataAsVectorArray(filetestimages, test_images, 10000, 28*28);
    readLabelDataAsVectorArray(filetestlabels, test_labels, 10000, 10);
    printf("reading finished!\n");

    /*
    int layers = 3;
    int sizeArray[3] = {4, 5, 3};
    init(sizeArray, layers);
    int i;
    for (i = 0; i < num_layers-1; i++) {
    	printf("%d<->%d:\n", i+1, i+2);
    	printf("weights:\n");
    	printMatrix(weights[i]);
    	printf("biases:\n");
    	printMatrix(biases[i]);
    	printf("\n");
    }

    matrix *input = newMatrix(4,1);
    for (i=0; i < 4; i++) {
    	setElement(input, i+1, 1, 1.);
    }
    matrix *output = feedforward(input);
    printf("feedforward......\n");
    printMatrix(output);
	*/

	/*
    int *array = (int*)malloc(60000*sizeof(int));
    randomShuffle(array, 60000);
    for (int i = 0; i < 60000; i++)
    	printf("%d ", array[i]);
    printf("\n");
	*/

    int layers = 3;
    int sizes[3] = {784, 30, 10};
    
    int success = init(sizes, layers);
    if (success == -1) {
    	printf("too many layers\n");
    	return 0;
    }
    SGD(train_images, train_labels, 6000, 30, 10, 0.5, test_images, test_labels, 1000, false);

    int i;
    for (i = 0; i < 60000; i++) {
        deleteMatrix(train_images[i]);
        deleteMatrix(train_labels[i]);
    }
    for (i = 0; i < 10000; i++) {
        deleteMatrix(test_images[i]);
        deleteMatrix(test_labels[i]);
    }
    return 0;
}