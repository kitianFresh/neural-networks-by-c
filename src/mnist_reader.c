#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "matrix.h"

char **strsplit(const char* str, const char* delim, size_t* numtokens) {
    // copy the original string so that we don't overwrite parts of it
    // (don't do this if you don't need to keep the old line,
    // as this is less efficient)
    //char *s = strdup(str);
    char *s = str;
    // these three variables are part of a very common idiom to
    // implement a dynamically-growing array
    size_t tokens_alloc = 1;
    size_t tokens_used = 0;
    char **tokens = calloc(tokens_alloc, sizeof(char*));
    char *token, *strtok_ctx;
    for (token = strtok_r(s, delim, &strtok_ctx);
            token != NULL;
            token = strtok_r(NULL, delim, &strtok_ctx)) {
        // check if we need to allocate more space for tokens
        if (tokens_used == tokens_alloc) {
            tokens_alloc *= 2;
            tokens = realloc(tokens, tokens_alloc * sizeof(char*));
        }
        tokens[tokens_used++] = strdup(token);
    }
    // cleanup
    if (tokens_used == 0) {
        free(tokens);
        tokens = NULL;
    } else {
        tokens = realloc(tokens, tokens_used * sizeof(char*));
    }
    *numtokens = tokens_used;
    //free(s); no longer need, because I don't use strdup
    return tokens;
}

//　给label用
matrix * vectorize(int i, int vectorlen) {
    matrix *v = newMatrix(vectorlen, 1);
    setElement(v, i+1, 1, 1.0);
    return v;
}

int readImageDataAsVectorArray(char *filename, matrix *images[], int num, int vectorlen) {
    if (filename == NULL || images == NULL || num <= 0 || vectorlen <= 0) return -1;
    matrix *image_vector;
    image_vector = newMatrix(vectorlen, 1);
    char *line = NULL;
    size_t linelen;
    char **tokens;
    size_t numtokens;
    int lno = 0;
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == NULL) exit(EXIT_FAILURE);

    while (getline(&line, &linelen, fp) != -1) {
  
        tokens = strsplit(line, ", \t\n", &numtokens);
        assert (numtokens == vectorlen);
        // 设置一列的元素
        for (size_t i = 0; i < numtokens; i++) {
            
            setElement(image_vector, i+1, 1, strtod(tokens[i], NULL));
            free(tokens[i]);
        }
        images[lno++] = copyMatrix(image_vector);
        if (tokens != NULL)
            free(tokens);
    }
    assert (lno == num);
        
    if (line != NULL) free(line);
    if (image_vector != NULL) deleteMatrix(image_vector);
    return 0;
}

int readLabelDataAsVectorArray(char *filename, matrix *labels[], int num, int vectorlen) {
    if (filename == NULL || labels == NULL || num <= 0 || vectorlen <= 0) return -1;
    char *line = NULL;
    size_t linelen;
    int lno = 0;
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == NULL) exit(EXIT_FAILURE);

    while (getline(&line, &linelen, fp) != -1) {       
        labels[lno++] = vectorize(atoi(line), vectorlen);
    }
    assert (lno == num);
        
    if (line != NULL) free(line);
    return 0;
}
