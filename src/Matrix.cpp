#include "Matrix.h"

Matrix::Matrix (int m, int n) : m(m), n(n){

}

void Matrix::initialElements() {
    elements = new float[n * m];
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            elements[i * n + j] = 1.;
        }
    }
}