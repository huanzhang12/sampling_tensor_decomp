#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "engine.h"

#include "matlab_wrapper.h"
#include "util.h"

Matlab_wrapper::Matlab_wrapper() {
	ep = engOpen(NULL);
	assert(ep);
}

Matlab_wrapper::~Matlab_wrapper() {
	engClose(ep);
}

// A: double[n][n]
// U, V: double[n][k]
// S: double[k]
void Matlab_wrapper::svds(double* A, int n, int k, double* U, double* S, double* V) {

	mxArray *mxA, *mxU, *mxS, *mxV;

	printf("Starting svds for %d x %d matrix with top %d vectors ...\n", n, n, k);

	// copying data
	mxA = mxCreateDoubleMatrix(n, n, mxREAL);
	for(int i = 0; i < n; i++) {
		memcpy(mxGetPr(mxA) + i * n, A + i * n, n * sizeof(double));
	}

	printf("Copying done.\n");

	char cmd[100];
	memset(cmd, 0, sizeof(cmd));
	sprintf(cmd, "[U, S, V] = svds(A, %d);", k);
	engPutVariable(ep, "A", mxA);
	engEvalString(ep, cmd);

	// reading data
	printf("Reading data ...\n");
	mxU = engGetVariable(ep, "U");
	mxS = engGetVariable(ep, "S");
	mxV = engGetVariable(ep, "V");
	printf("finish reading data ...\n");
    if (U) {
	for(int i = 0; i < n; i++)
		for(int j = 0; j < k; j++) {
			U[i*k+j] = *(mxGetPr(mxU) + j * n + i);
		}
    }
    if (V) {
    for(int i = 0; i < n; i++)
        for(int j = 0; j < k; j++) {
            V[i*k+j] = *(mxGetPr(mxV) + j * n + i);
        }
    }
    if (S) {
	for(int i = 0; i < k; i++)
		S[i] = *(mxGetPr(mxS) + i * k + i);
    }
	printf("start to release memory...\n");
	// close engine and free memory
	mxDestroyArray(mxA);
	mxDestroyArray(mxU);
	mxDestroyArray(mxS);
	mxDestroyArray(mxV);

}

void Matlab_wrapper::eigs(double* A, int n, int k, double* U, double* S) {

	mxArray *mxA, *mxU, *mxS;

	printf("Starting eigs for %d x %d matrix with top %d vectors ...\n", n, n, k);

	// copying data
	mxA = mxCreateDoubleMatrix(n, n, mxREAL);
	for(int i = 0; i < n; i++) {
		memcpy(mxGetPr(mxA) + i * n, A + i * n, n * sizeof(double));
	}

	printf("Copying done.\n");

	char cmd[100];
	memset(cmd, 0, sizeof(cmd));
	sprintf(cmd, "[U, S] = eigs(A, %d);", k);
	engPutVariable(ep, "A", mxA);
	engEvalString(ep, cmd);

	// reading data
	printf("Reading data ...\n");
	mxU = engGetVariable(ep, "U");
	mxS = engGetVariable(ep, "S");
    if (U) {
	for(int i = 0; i < n; i++)
		for(int j = 0; j < k; j++) {
			U[i*k+j] = *(mxGetPr(mxU) + j * n + i);
		}
    }
    if (S) {
	for(int i = 0; i < k; i++)
		S[i] = *(mxGetPr(mxS) + i * k + i);
    }

	// close engine and free memory
	mxDestroyArray(mxA);
	mxDestroyArray(mxU);
	mxDestroyArray(mxS);    

}

mxArray* Matlab_wrapper::create_matrix(double* A, int m, int n, bool transpose) {

	mxArray* mxA;

	if (!transpose) {
		mxA = mxCreateDoubleMatrix(m, n, mxREAL);
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				*(mxGetPr(mxA) + j * m + i) = A[IND2D(i,j,n)];
	}
	else {
		mxA = mxCreateDoubleMatrix(n, m, mxREAL);
		for(int i = 0; i < m; i++)
			for(int j = 0; j < n; j++)
				*(mxGetPr(mxA) + i * n + j) = A[IND2D(i,j,n)];
	}

	return mxA;

}

void Matlab_wrapper::multiply(double *A, double* B, int m1, int n1, int m2, int n2, int m3, int n3, bool transposeA, bool transposeB, double* C) {

	mxArray *mxA, *mxB, *mxC;

	// sanity check
	int t1 = m1, t2 = n1, t3 = m2, t4 = n2, t;
	if (transposeA) {t = t1; t1 = t2; t2 = t;}
	if (transposeB) {t = t3; t3 = t4; t4 = t;} 
	assert(t2 == t3);
	assert(t1 == m3);
	assert(t4 == n3);

	mxA = create_matrix(A, m1, n1, transposeA);
	mxB = create_matrix(B, m2, n2, transposeB);

	engPutVariable(ep, "A", mxA);
	engPutVariable(ep, "B", mxB);

	engEvalString(ep, "C = A * B;");

	mxC = engGetVariable(ep, "C");
	for(int i = 0; i < m3; i++)
		for(int j = 0; j < n3; j++)
			C[IND2D(i,j,n3)] = *(mxGetPr(mxC) + j * m3 + i);

	mxDestroyArray(mxA);
	mxDestroyArray(mxB);
	mxDestroyArray(mxC);

}

void Matlab_wrapper::pinv(double* A, int m, int n, double* Ainv) {

	mxArray *mxA, *mxAinv;

	mxA = create_matrix(A, m, n, false);

	engPutVariable(ep, "A", mxA);
	engEvalString(ep, "Ainv = pinv(A);");
	mxAinv = engGetVariable(ep, "Ainv");

	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			Ainv[IND2D(i,j,m)] = *(mxGetPr(mxAinv) + j * n + i);

	mxDestroyArray(mxA);
	mxDestroyArray(mxAinv);

}

// A: m x n
// x: n x 1
// y: m x 1
void Matlab_wrapper::mldivide(double* A, double* y, int m, int n, bool transposeA, double* x) {

    mxArray *mxA, *mxX, *mxY;
    
    mxA = create_matrix(A, m, n, transposeA);
    if (transposeA) {
        int t = m; m = n; n = t;
    }
    
    mxY = mxCreateDoubleMatrix(m, 1, mxREAL);
    memcpy(mxGetPr(mxY), y, sizeof(double) * m);
    
    engPutVariable(ep, "A", mxA);
    engPutVariable(ep, "y", mxY);
    engEvalString(ep, "x = A \\ y;");
    mxX = engGetVariable(ep, "x");
    
    memcpy(x, mxGetPr(mxX), sizeof(double) * n);
    
    mxDestroyArray(mxA);
    mxDestroyArray(mxX);
    mxDestroyArray(mxY);

}

void Matlab_wrapper::drchrnd(int n, double* alpha, double* ret) {

	mxArray *mxAlpha, *mxRet;

	mxAlpha = mxCreateDoubleMatrix(1, n, mxREAL);
	memcpy(mxGetPr(mxAlpha), alpha, sizeof(double) * n);
	engPutVariable(ep, "alpha", mxAlpha);

	char cmd[1000];
	memset(cmd, 0, sizeof(cmd));
	sprintf(cmd, "ret = gamrnd(alpha, ones(1, %d));", n);
	engEvalString(ep, cmd);
	mxRet = engGetVariable(ep, "ret");
    assert(mxRet);

	memcpy(ret, mxGetPr(mxRet), sizeof(double) * n);

	// normalize
	double sum = 0;
	for(int i = 0; i < n; i++)
		sum += ret[i];
	double scale = 1.0 / sum;
	for(int i = 0; i < n; i++)
		ret[i] *= scale;

}

void Matlab_wrapper::mnrnd(int n, double* p, int m, int* ret) {

	// sanity check
	double sum = 0;
	for(int i = 0; i < m; i++) {
		assert(0 <= p[i] && p[i] <= 1);
		sum += p[i];
	}
	assert(safe_compare(sum, 1.0) == 0);

	mxArray *mxP, *mxRet;

	mxP = mxCreateDoubleMatrix(1, m, mxREAL);
	memcpy(mxGetPr(mxP), p, sizeof(double) * m);
	engPutVariable(ep, "p", mxP);

	char cmd[1000];
	memset(cmd, 0, sizeof(cmd));
	sprintf(cmd, "ret = mnrnd(%d, p);", n);
	engEvalString(ep, cmd);
	mxRet = engGetVariable(ep, "ret");

	for(int i = 0; i < m; i++)
		ret[i] = lround(*(mxGetPr(mxRet) + i));

	// sanity check
	int t = 0;
	for(int i = 0; i < m; i++)
		t += ret[i];
	assert(t == n);
	
	mxDestroyArray(mxP);
	mxDestroyArray(mxRet);

}

void Matlab_wrapper::lognrnd(int n, double mu, double sigma, double* ret) {

    // sanity check
    assert(sigma > 0);
    
    mxArray *mxRet;
    
    char cmd[1000];
    memset(cmd, 0, sizeof(cmd));
    sprintf(cmd, "ret = lognrnd(%lf, %lf, %d, 1);", mu, sigma, n);
    engEvalString(ep, cmd);
    mxRet = engGetVariable(ep, "ret");
    
    memcpy(ret, mxGetPr(mxRet), sizeof(double) * n);
    
    mxDestroyArray(mxRet);

}

// C: real[dim1][len]
// d: real[dim1]
// A: real[dim2][len], B: real[dim2][1]
// Aeq: real[dim3][len], Beq: real[dim3]
// lb, ub: real[len][1]
void Matlab_wrapper::lsqlin(int len, int dim1, int dim2, int dim3, double* C, double* d, double* A, double* B, double* Aeq, double* Beq, double* lb, double* ub, double* x) {

    assert(len > 0 && dim1 > 0);
    assert(dim2 >= 0 && dim3 >= 0 && (dim2 > 0 || dim3 > 0));

    mxArray* mxC = create_matrix(C, dim1, len, false);
    mxArray* mxd = create_matrix(d, dim1, 1, false);
    mxArray* mxlb = create_matrix(lb, len, 1, false);
    mxArray* mxub = create_matrix(ub, len, 1, false);

    engPutVariable(ep, "C", mxC);
    engPutVariable(ep, "d", mxd);
    engPutVariable(ep, "lb", mxlb);
    engPutVariable(ep, "ub", mxub);
    
    if (dim2 == 0) {
        // only equality constraints        
        mxArray* mxAeq = create_matrix(Aeq, dim3, len, false);
        mxArray* mxBeq = create_matrix(Beq, dim3, 1, false);
        engPutVariable(ep, "Aeq", mxAeq);
        engPutVariable(ep, "Beq", mxBeq);
        engEvalString(ep, "x = lsqlin(C, d, [], [], Aeq, Beq, lb, ub);");
        mxArray* mxX = engGetVariable(ep, "x");
        memcpy(x, mxGetPr(mxX), sizeof(double) * len);
        mxDestroyArray(mxAeq);
        mxDestroyArray(mxBeq);
        mxDestroyArray(mxX);
    }
    else {
        assert(0);
    }
    
    mxDestroyArray(mxC);
    mxDestroyArray(mxd);
    mxDestroyArray(mxlb);
    mxDestroyArray(mxub);

}

/*
int main() {

	double A[2][3] = {{1, 2, 3}, {4, 5, 6}};
	double B[3] = {3, 3, 3};
	double C[2];
	
	printf("I am here\n");
	
	Matlab_wrapper* mat_wrapper = new Matlab_wrapper();
	
	printf("matlab initialized\n");

	mat_wrapper->mldivide(reinterpret_cast<double*>(A), C, 2, 3, false, B);
	printf("%lf %lf %lf\n", B[0], B[1], B[2]);
    
    mat_wrapper->mldivide(reinterpret_cast<double*>(A), B, 2, 3, true, C);
    printf("%lf %lf\n", C[0], C[1]);

	puts("=================================================================");


	return 0;

}*/
