#
# Makefile for Test Programs
#

CC = gcc
CFLAGS  = -g -Wall
RM = rm
RM_FLAGS = -f

default: test_reader test_matrix

#test_reader: test_reader.o
#	$(CC) $(CFLAGS) -I../include -o test_reader test_reader.o

test_reader: mnist_reader.o matrix.o 
	$(CC) $(CFLAGS) -I../include -o test_reader test_reader.c mnist_reader.o matrix.o -lm
	
test_matrix : matrix.o
	$(CC) $(CFLAGS) -I../include -o test_matrix test_matrix.c matrix.o -lm


matrix.o: ../src/matrix.c ../include/matrix.h
	$(CC) $(CFLAGS) -I../include -c ../src/matrix.c -lm

mnist_reader.o: ../src/matrix.c ../src/mnist_reader.c ../include/matrix.h ../include/mnist_reader.h
	$(CC) $(CFLAGS) -I../include -c ../src/mnist_reader.c ../src/matrix.c -lm

clean:
	$(RM) $(RM_FLAGS)  *.o
