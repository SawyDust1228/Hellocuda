#ifndef MATRIX_H
#define MATRIX_H

struct Matrix
{
    int m;
    int n;
    float* elements;
    Matrix (int m, int n);

    void initialElements();
};


#endif




