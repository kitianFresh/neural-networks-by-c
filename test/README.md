# Test

对矩阵操作 `matrix.c` 的测试代码, `gcc -I../include test_matrix.c  ../src/matrix.c -o test_matrix  -lm`;
对矩阵操作 `mnist_reader.c` 的测试代码, `gcc -I../include test_reader.c ../src/mnist_reader.c ../src/matrix.c -o test_reader -lm`;

## 错误
    1. Make 要求每一个command 之前必须加上 tab, 但是Make不认识 4个space组成的tab, 因此, 使用 VSCode 编辑的无法运行. `Makefile:21: *** missing separator.  Stop.`
    2. Make 编写过程中的依赖, 需要指明是哪一个目录下的, 要不然 Make 找不到这个文件. 可以直接写 路径全名, 也可以设置 VPATH


