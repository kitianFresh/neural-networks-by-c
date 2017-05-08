#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"
#include "mnist_reader.h"
#include <stdbool.h>

#define MAX_LAYER 8

int num_layers;					//层数
int sizes[MAX_LAYER];			//每一层单元数
matrix *biases[MAX_LAYER-1];  	//层间biases
matrix *weights[MAX_LAYER-1]; 	//层间weights


int contains(int *array, int size, int elem) {
	if (size <= 0) return 0;
	int i;
	for (i = 0; i < size; i++) {
		if (array[i] == elem){
			return 1;
		}
	}
	return 0;
}
//产生一个随机混洗的数组下标
int randomShuffle(int *array, int size) {
	srand(time(NULL));
	int i;
	int bad;
	for (i = 0; i < size; i ++) {
		bad = 1;
		while (bad) {
			array[i] = rand() % size;
			//printf("array[%d]: %d\n", i, array[i]);
			if (i >0 && contains(array, i, array[i])) bad = 1;
			else bad = 0;
		}
	}
	return 0;
}

int init(int *sizeArray, int n_layers) {
	if (n_layers > MAX_LAYER || sizeArray == NULL) return -1;
	num_layers = n_layers;
	
	int i;
	for (i = 0; i < n_layers; i++)
		sizes[i] = sizeArray[i];

	for (i = 0; i < num_layers-1; i++) {
		matrix *m = randnMatrix(sizes[i+1], sizes[i], 0, 1);
		matrix *v = randnMatrix(sizes[i+1], 1, 0, 1);
		weights[i] = m;
		biases[i] = v;
	}
	return 0;
}

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prime(double x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

matrix * cost_derivative(matrix *output_activation, matrix *y) {
	matrix *result = newMatrix(y->rows, y->cols);
	minus(output_activation, y, result);
	return result;
}

matrix *feedforward(matrix *in) {
	if (in == NULL) return NULL;

	matrix *a = copyMatrix(in);
	int i;
	for (i = 0; i < num_layers-1; i++) {
		matrix *temp = newMatrix(weights[i]->rows, a->cols);
		product(weights[i], a, temp);
		sum(temp, biases[i], temp);
		funcMatrix(temp, sigmoid);
		deleteMatrix(a); //及时释放上一次的temp，以免占用内存过多,发生内存泄露
		a = temp;
	}
	return a;
}

int backprop(matrix *x, matrix *y, matrix *nabla_w[], matrix *nabla_b[]) {
	int i;
	
	//feedforward
	matrix *activation = copyMatrix(x);
	matrix **activations = (matrix**)malloc(num_layers * sizeof(char*));//存储所有的激活层
	matrix **zs = (matrix**)malloc(num_layers * sizeof(char*));			//存储所有的加权层
	activations[0] = activation;
	//printf("backprop: testtttttt0\n");
	for (i = 0; i < num_layers-1; i++) {
		matrix *z = newMatrix(weights[i]->rows, activation->cols);
		product(weights[i], activation, z); // z = W*a
		sum(z, biases[i], z); 				// z = W*a + b
		zs[i+1] = copyMatrix(z);			// zs.add(z)zs[1]zs[2]..zs[num_layers-1]对应2,3...L层,zs[0]不用，因为没有
		funcMatrix(z, sigmoid);				// 
		activation = z;						// a = sigmoid(z)
		activations[i+1] = activation;		// as.add(a)as[0]as[1]..as[num_layers-1]对应1,2...L层
	}
	//printf("backprop: testtttttt1\n");
	//backward pass
	//首先可以直接根据y算出最后一层的delta
	matrix *nabla_c = cost_derivative(activations[num_layers-1], y);
	matrix *nabla_z = copyMatrix(zs[num_layers-1]);
	funcMatrix(nabla_z, sigmoid_prime);
	matrix *delta = newMatrix(nabla_z->rows, nabla_z->cols); 
	scalarProduct(nabla_c, nabla_z, delta);	 // deltaL = nabla_C (*) sigmoid_prime(z)
	deleteMatrix(nabla_c);
	deleteMatrix(nabla_z);
	//printf("backprop: testtttttt2\n");
	nabla_b[num_layers-2] = delta;
	transposeSelf(activations[num_layers-2]); // 倒数第二层的激活的转置 a(L-1).T
	//printf("backprop: testtttttt3\n");//num_layers-2才是L-1<>L之间的权值矩阵，即最后一个权值矩阵
	matrix *dp = newMatrix(weights[num_layers-2]->rows, weights[num_layers-2]->cols);
	//printf("backprop: testtttttt4\n");
	product(delta, activations[num_layers-2], dp); // delta product a(L-1).T
	nabla_w[num_layers-2] = dp;

	//printf("backprop: testtttttt5\n");

	int l;
	// 从倒数第二层开始反向传播计算每一层的delta，并保存每一层的偏导nabla_b 和nabla_w
	// 注意！
	// 对于zs[0,1,2...num_layers-1]分别对应1,2,3...L层的z，但是第一层没有z，所以不用
	// 对于as[0,1,2...num_layers-1]分别对应1,2,3...L层的a，没一层都有，第一层就是输入自己
	// 对于weights[0,1,2...num_layers-2]分别对应1<>2, 2<>3... L-1<>L之间的权值矩阵
	// 对于 biases[0,1,2...num_layers-2]分别对应1<>2, 2<>3... L-1<>L之间的偏差矩阵
	// 对于层数，是按照下标从0到num_layers-1的，分别对应1,2,3...L，这里的l从L-1开始递减到1
	for (l = num_layers-2; l >= 1; l--) {
		matrix *sp = copyMatrix(zs[l]);
		funcMatrix(sp, sigmoid_prime);
		matrix *wt = copyMatrix(weights[l]);
		//printf("backprop: testtttttt6\n");
		transposeSelf(wt);
		matrix *deltal = newMatrix(biases[l-1]->rows, biases[l-1]->cols);
		int ret = product(wt, nabla_b[l], deltal);
		if (ret == -2) {
			printf("wrong size: wt size: (%d, %d) nabla_b[%d] size: (%d, %d), deltal size: (%d, %d)\n", 
				wt->rows, wt->cols, l, nabla_b[l]->rows, nabla_b[l]->cols, deltal->rows, deltal->cols);
		}
		scalarProduct(deltal, sp, deltal);	// delta[l] = (w[l+1].T * delta[l+1]) (*) sp(zl)
		//printf("backprop: testtttttt7\n");
		nabla_b[l-1] = deltal;
		//printf("nabla_b[%d]\n", l-1);
		//printMatrix(nabla_b[l-1]);
		//printf("backprop: testtttttt8\n");
		//printMatrix(activations[l-1]);
		transposeSelf(activations[l-1]); //前面一层的激活的转置 a(l-1).T
		//printf("backprop: testtttttt9\n");
		matrix *dp = newMatrix(weights[l-1]->rows, weights[l-1]->cols);
		product(deltal, activations[l-1], dp); // deltal product a(l-1).T
		nabla_w[l-1] = dp;

		deleteMatrix(sp); //sp是一个临时矩阵，及时释放掉，以免内存泄露
		deleteMatrix(wt);
	}
	//printf("backprop: testtttttt10\n");
	
	// 释放所有的zs和activations, 已经用不到了
	for (l = 1; l < num_layers; l++) {
		deleteMatrix(zs[l]);
		deleteMatrix(activations[l]);
	}
	deleteMatrix(activations[0]);
	free(zs);
	free(activations);
	return 0;
}


int evaluate(matrix *test_images[], matrix *test_labels[], int test_size) {
	int i, count;
	matrix *output;
	count = 0;
	for (i = 0; i < test_size; i++) {
		output = feedforward(test_images[i]);
		count += maxVector(output) == maxVector(test_labels[i]) ? 1 :  0;
		deleteMatrix(output);//feedforward会生成新矩阵
	}
	return count;
}

double square(double x) {
	return x * x;
}

double evaluate_regression(matrix *test_datas[], matrix *test_labels[], int size) {
	matrix * output;
	matrix * result = newMatrix(test_labels[0]->rows, test_labels[0]->cols);
	double err, v1, v2;
	err = 0.0;
	for (int i = 0; i < size; i++) {
		output = feedforward(test_datas[i]);
		v1 = getElement(test_labels, 1, 1);
		v2 = getElement(output, 1, 1);
		err +=  (v1 - v2)*(v1 - v2);
		
	}
	deleteMatrix(output);
	return sqrt(err);
}

int update_mini_batch(matrix *mini_batch_images[], matrix *mini_batch_labels[], int mini_batch_size, double eta) {
	int i;
	// 初始化矩阵数组，用于存放累deltaC的累加值，因为是batch
	matrix **nabla_w = (matrix**)malloc((num_layers-1) * sizeof(char*));
	matrix **nabla_b = (matrix**)malloc((num_layers-1) * sizeof(char*));
	for (i = 0; i < num_layers-1; i++) {
		nabla_w[i] = newMatrix(weights[i]->rows, weights[i]->cols);
		nabla_b[i] = newMatrix(biases[i]->rows, biases[i]->cols);
	}

	//minibatch sum
	matrix **delta_nabla_w = (matrix**)malloc((num_layers-1) * sizeof(char*));
	matrix **delta_nabla_b = (matrix**)malloc((num_layers-1) * sizeof(char*));
	for (i = 0; i < mini_batch_size; i++) {
		//注意！！不需要对delta_nabla_w[i]和delta_nabla_b[i]初始化空间，因为backprop内部会分配空间给他们；
		/*
		for (int j = 0; j < num_layers-1; j++) {
			delta_nabla_w[j] = newMatrix(weights[j]->rows, weights[j]->cols);
			delta_nabla_b[j] = newMatrix(biases[j]->rows, biases[j]->cols);
		}
		*/
		// 对一个x，y反向传播一次
		backprop(mini_batch_images[i], mini_batch_labels[i], delta_nabla_w, delta_nabla_b);
		
		for (int j = 0; j < num_layers-1; j++) {
			sum(nabla_w[j], delta_nabla_w[j], nabla_w[j]);
			sum(nabla_b[j], delta_nabla_b[j], nabla_b[j]);
		}

		// 运行完一次backprop，delta_nabla_w 和delta_nabla_b便不再使用，及时释放
		
		for(int j = 0; j < num_layers-1; j++){
			deleteMatrix(delta_nabla_w[j]);
			deleteMatrix(delta_nabla_b[j]);
		}

	}

	//释放delta_nabla_w和delta_nabla_b指针数组自己
	free(delta_nabla_b);
	free(delta_nabla_w);

	// 更新w和b
	for (i = 0; i < num_layers-1; i++) {
		multiplyMatrix(nabla_w[i], eta/mini_batch_size);
		minus(weights[i], nabla_w[i], weights[i]);

		multiplyMatrix(nabla_b[i], eta/mini_batch_size);
		minus(biases[i], nabla_b[i], biases[i]);
	}

	//释放空间
	for (i = 0; i < num_layers-1; i++) {
		deleteMatrix(nabla_w[i]);
		deleteMatrix(nabla_b[i]);
	}
	free(nabla_w);
	free(nabla_b);

}


int SGD(matrix *train_images[], matrix *train_labels[], int train_size, int epochs, int mini_batch_size, double eta,
		matrix *test_images[], matrix *test_labels[], int test_size, bool regression) {
	
	matrix **mini_batch_images = (matrix**)malloc(mini_batch_size * sizeof(char*));
	matrix **mini_batch_labels = (matrix**)malloc(mini_batch_size * sizeof(char*));
	int i,k;
	int *array = (int*)malloc(train_size * sizeof(int));
	for (i = 1; i <= epochs; i++) {
		//混洗，即改变原来train_images的数组指针的指向
		randomShuffle(array, train_size);
		/*
		for (k = 0; k < train_size; k++) {
			train_images[k] = train_images[array[k]];
			train_labels[k] = train_labels[array[k]];
		}
		*/
		//以上是错误的，交换过程中会使得元素相互覆盖了，其实可以不改变原来的顺序，
		//我们只是在访问train_images的时候按照混洗顺序访问即可	
		for (int start = 0; start < train_size; start+=mini_batch_size) {
			for(k = 0; k < mini_batch_size; k++) {
				mini_batch_images[k] = train_images[array[start+k]];
				mini_batch_labels[k] = train_labels[array[start+k]];
			}
			//printf("SGD: testtttttt2\n");
			update_mini_batch(mini_batch_images, mini_batch_labels, mini_batch_size, eta);
			//printf("SGD: testtttttt3\n");
		}

		if (test_images != NULL && test_labels != NULL) {
			if (regression) {
				double err = evaluate_regression(test_images, test_labels, test_size);
				printf("Epoch %d: rmsd: %lf\n", i, err);
			}
			else {
				int corrects = evaluate(test_images, test_labels, test_size);
				printf("Epoch %d: %d / %d %lf\n", i, corrects, test_size, 1.0*corrects/test_size);
			}
			
		}
		else {
			printf("Epoch %d complete\n", i);
		}
	}


	free(array);
	free(mini_batch_images);
	free(mini_batch_labels);

	return 0;
}

int predict(matrix * datas[], int size, matrix * y_hat[]) {
	matrix * output;
	for (int i = 0; i < size; i++) {
		output = feedforward(datas[i]);
		y_hat[i] = output;
		deleteMatrix(output);
	}
}