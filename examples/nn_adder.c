#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <math.h>

#define HIDDEN_SIZE 10
#define INPUT_SIZE 2
#define TRAINING_SIZE 10000
#define EPOCH 50

double w[HIDDEN_SIZE][INPUT_SIZE];
double v[HIDDEN_SIZE];
double A = 0.005;
double o[HIDDEN_SIZE];

double data0_min, data0_max, data1_min, data1_max, label_min, label_max;

double randomer(double left, double right) {
    return (double) rand() / (RAND_MAX) * (right - left) + left;
}

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

void init_weights() {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            w[i][j] = randomer(-1, 1);
        }
        v[i] = randomer(-1, 1);
        o[i] = randomer(-1, 1);
        // printf("%lf, %lf, %lf, %lf\n", w[i][0], w[i][1], v[i], o[i]);
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double forward(double x0, double x1) {
    double y = 0;
    for (int i=0; i<HIDDEN_SIZE; i++) {
        o[i] = sigmoid(x0 * w[i][0] + x1 * w[i][1]);
        y += o[i] * v[i];
    }
    // y = sigmoid(y);
    return y;
}

// y_ 为标签, y为预测输出
double backward(double x0, double x1, double y, double y_) {
    // double dv = y * (1 - y) * (y_ - y);
    double dv = (y_ - y);
    // 最后一层
    for (int i=0; i<HIDDEN_SIZE; i++) { // 更新 v[]
        v[i] += A * dv * o[i];
    }
    // 前一层
    for (int i=0; i<HIDDEN_SIZE; i++) { // 更新 w[][]
        w[i][0] += A * o[i] * (1 - o[i]) * dv * v[i] * x0;
        w[i][1] += A * o[i] * (1 - o[i]) * dv * v[i] * x1;
    }
}

int generate_data(double (*datas)[2], double * labels, int left, int right, int size) {
    for (int i=0; i<size; i++) {
        datas[i][0] = randomer(left, right);
        datas[i][1] = randomer(left, right);
        labels[i] = datas[i][0] + datas[i][1];

        // printf("%lf + %lf = %lf\n", datas[i][0], datas[i][1], labels[i]);
    }
}

double fit(double (*datas)[2], double *labels, int train_size, double (*test)[2], double *test_label, int test_size) {
    int y;
    double train_rmse = 0.0;
    double test_rmse = 0.0;
    int * array = (int*)malloc(sizeof(int) * train_size);
    
    for (int i = 0;  i < EPOCH; i++) {
        randomShuffle(array, train_size);
        // SGD
        for (int i = 0; i < train_size; i++) {
            y = forward(datas[array[i]][0], datas[array[i]][1]);
            backward(datas[array[i]][0], datas[array[i]][1], y, labels[array[i]]);
        }

        // 计算误差 TrainSet RMSE
        for (int i = 0; i < train_size; i++) {
            y = forward(datas[i][0], datas[i][1]);
            train_rmse += (y - labels[i]) * (y - labels[i]);
        }
        train_rmse = sqrt(train_rmse/train_size);

        // 计算误差 TestSet RMSE
        for (int i = 0; i < test_size; i++) {
            y = forward(test[i][0], test[i][1]);
            test_rmse += (y - test_label[i]) * (y - test_label[i]);
        }
        test_rmse = sqrt(test_rmse/test_size);
        printf("epoch %d, test rmse %lf, test_rmse %lf\n", i, train_rmse, test_rmse);
    }
    
}


void norm(double (*datas)[2], double *labels, int train_size) {
    data0_min = data0_max = datas[0][0];
    data1_min = data1_max = datas[0][1];
    label_min = label_max = labels[0];
    for (int i = 1; i < train_size; i++) {
        if (datas[i][0] > data0_max) {
            data0_max = datas[i][0];
        }
        if (datas[i][0] < data0_min) {
            data0_min = datas[i][0];
        }

        if (datas[i][1] > data1_max) {
            data1_max = datas[i][1];
        }
        if (datas[i][1] < data1_min) {
            data1_max = datas[i][1];
        }
        
        if (labels[i] > label_max) {
            label_max = labels[i];
        }
        if (labels[i] < label_min) {
            label_min = labels[i];
        }
    }
    for (int i = 0; i < train_size; i++) {
        datas[i][0] = (datas[i][0] - data0_min + 1) / (data0_max - data0_min + 1);
        datas[i][1] = (datas[i][1] - data1_min + 1) / (data1_max - data1_min + 1);
        // labels[i] = (labels[i] - label_min + 1) / (label_max - label_min + 1);
        // printf("norm: %lf, %lf, %lf\n", datas[i][0], datas[i][1], labels[i]);
    }
}

double predict(double x0, double x1) {
    x0 = (x0 - data0_min + 1) / (data0_max - data0_min + 1);
    x1 = (x1 - data1_min + 1) / (data1_max - data1_min + 1);
    return forward(x0, x1);
    // return forward(x0, x1) * (label_max - label_min + 1) + label_min - 1;
}

int main() {
    init_weights();
    double datas[TRAINING_SIZE][2];
    double labels[TRAINING_SIZE];
    double tests[TRAINING_SIZE/5][2];
    double t_labels[TRAINING_SIZE/5];
    generate_data(datas, labels, 0, 10, TRAINING_SIZE);
    generate_data(tests, t_labels, 0, 10, TRAINING_SIZE/5);

    norm(datas, labels, TRAINING_SIZE);
    norm(datas, labels, TRAINING_SIZE/5);
    fit(datas, labels, TRAINING_SIZE, tests, t_labels, TRAINING_SIZE/5);

    double x0, x1;
    while (1) {
        scanf("%lf, %lf", &x0, &x1);
        printf("%lf\n", predict(x0, x1));
    }
    return 0;
}
