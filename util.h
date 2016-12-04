#ifndef UTIL_H_
#define UTIL_H_

#include <math.h>
#include "fftw3.h"

#define PI (3.14159265359)

#define SQR(x) ((x)*(x))
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

#define IND2D(i,j,n) ((i)*(n)+(j))
#define IND3D(i,j,k,n) (((size_t)i)*((size_t)n)*((size_t)n)+((size_t)j)*((size_t)n)+((size_t)k))

#define POWER2(b) (1<<(b))
#define MASK2(b) ((1<<(b))-1)

inline int safe_compare(double a, double b) {
	if (fabs(a-b) <= 1e-7) return 0;
	else if (a < b) return -1;	
	else return 1;
}

inline int safe_compare(double a, double b, double eps) {
	if (fabs(a-b) <= eps) return 0;
	else if (a < b) return -1;
	else return 1;
}

inline void complex_assign(const fftw_complex a, fftw_complex b) {
	b[0] = a[0];
	b[1] = a[1];
}

inline void complex_conj(const fftw_complex a, fftw_complex b) {
	b[0] = a[0];
	b[1] = -a[1];
}

inline void complex_add(const fftw_complex a, const fftw_complex b, fftw_complex c) {
	c[0] = a[0] + b[0];
	c[1] = a[1] + b[1];
}

inline void complex_sub(const fftw_complex a, const fftw_complex b, fftw_complex c) {
	c[0] = a[0] - b[0];
	c[1] = a[1] - b[1];
}

inline void complex_mult(const fftw_complex a, const fftw_complex b, fftw_complex c) {
	fftw_complex t;
	t[0] = a[0] * b[0] - a[1] * b[1];
	t[1] = a[0] * b[1] + a[1] * b[0];
	c[0] = t[0];
	c[1] = t[1];
}

inline void complex_mult_conj(const fftw_complex a, const fftw_complex b, fftw_complex c) {
	fftw_complex t;
	t[0] = a[0] * b[0] + a[1] * b[1];
	t[1] = a[1] * b[0] - a[0] * b[1];
	c[0] = t[0];
	c[1] = t[1];
}

inline double complex_sqrnorm(const fftw_complex a) {
    return SQR(a[0]) + SQR(a[1]);
}

inline double dot_prod(double* a, double* b, int len) {
    double ret = 0;
    for(int i = 0; i < len; i++)
        ret += a[i] * b[i];
    return ret;
}

inline double vector_sqr_fnorm(double* a, int len) {
	double ret = 0;
	for(int i = 0; i < len; i++)
		ret += SQR(a[i]);
	return ret;
}

inline double compute_two_norm(double* a, double* b, int len) {
    double ret = 0;
    for(int i = 0; i < len; i++)
        ret += SQR(a[i] - b[i]);
    return ret;
}

inline double compute_one_norm(double* a, double* b, int len) {
	double ret = 0;
	for(int i = 0; i < len; i++)
		ret += fabs(a[i] - b[i]);
	return ret;
}

int compare_double(const void* p1, const void* p2);
int compare_int(const void* p1, const void* p2);
void print_complex_vector(const fftw_complex*, int);
double generate_std_normal();
void generate_uniform_sphere_point(int dim, double* ret);
void gram_schmidt_process(int n, int dim, double** x); // x: size n x dim
void sort_eigen_value_vector(double*, double**, int n, int dim);
int parse_cmd_arguments(int argc, char* argv[], int* para_groups, char** para_names, char** para_values);

int categorical_sampling(double* p, int d);
double gamma_sampling(double alpha); // sampling from Gamma(delta, 1)
void dirichlet_sampling(double* ret, double* alpha, int d); // sampling from Dirichlet(alpha)

#endif
