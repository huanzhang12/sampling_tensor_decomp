#ifndef MATLAB_WRAPPER_H_
#define MATLAB_WRAPPER_H_

#include "engine.h"

class Matlab_wrapper {

public:
	Matlab_wrapper();
	~Matlab_wrapper();
    
    Engine* get_engine() {
        return this->ep;
    }
    
	mxArray* create_matrix(double *A, int m, int n, bool transpose);    

	// A: double [n][n]
	// U, V: double [n][k]
	// S: double[k]
	void svds(double* A, int n, int k, double* U, double* S, double* V);
	
	// A: double [n][n]
	// U: double[n][k]
	// S: double[k] 
	void eigs(double* A, int n, int k, double* U, double* S);

	// A: double [m1][n1]
	// B: double [m2][n2]
	// C: returned matrix
	void multiply(double *A, double* B, int m1, int n1, int m2, int n2, int m3, int n3, bool transposeA, bool transposeB, double* C);

	// A: double [m][n]
	// Ainv: double [n][m]
	void pinv(double* A, int m, int n, double* Ainv);
    
    // solve Ax = y given A and y
    // A: double [m][n]
    // x: double [m] (if not transposed)
    // y: double [n] (if not transposed)
    void mldivide(double* A, double* y, int m, int n, bool transposeA, double* x);

	// generate m n-dimensional vectors from Dir(alpha)
	// alpha must have length n
	// ret: double [n]
	void drchrnd(int n, double* alpha, double* ret);

	// m choose n
	// p: double[m], must sum to one
	// ret: int[m], sum to n
	void mnrnd(int n, double* p, int m, int* ret);    
	
	// ret: int[n], i.i.d. generated from Lognormal(mu, sigma)
	void lognrnd(int n, double mu, double sigma, double* ret);
    
    // C: real[dim1][len]
    // d: real[dim1]
    // A: real[dim2][len], B: real[dim2][1]
    // Aeq: real[dim3][len], Beq: real[dim3]
    void lsqlin(int len, int dim1, int dim2, int dim3, double* C, double* d, double* A, double* B, double* Aeq, double* Beq, double* lb, double* ub, double* x);

private:
	Engine *ep;


};

#endif
