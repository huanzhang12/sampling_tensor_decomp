#ifndef FAST_TENSOR_POWER_METHOD_H_
#define FAST_TENSOR_POWER_METHOD_H_

#include "tensor.h"
#include "count_sketch.h"
#include "fft_wrapper.h"
#include "matlab_wrapper.h"

double fast_Tuuu(CountSketch* cs_T, CountSketch* f_cs_u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper);
void fast_TIuu(CountSketch* f_cs_T, CountSketch* f_cs_u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret);
void fast_TIuv(AsymCountSketch* f_cs_T, AsymCountSketch* f_cs_u2, AsymCountSketch* f_cs_u3, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret);
double fast_Tuvw(AsymCountSketch* f_cs_T, AsymCountSketch* f_cs_u1, AsymCountSketch* f_cs_u2, AsymCountSketch* f_cs_u3, int, FFT_wrapper* , FFT_wrapper* );
double fast_asym_Tuuu(CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper);
double fast_collide_Tuuu(CountSketch* cs_T, CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, double* u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper);
void fast_collide_TIuu(CountSketch* cs_T, CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, double* u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret);

double fast_sqr_fnorm(AsymCountSketch* cs_T);
double fast_sqr_fnorm(CountSketch* cs_T);

// lambda: double[rank]
// v: double[rank][dim]
// Note: after factorization cs_T and tensor will store the deflated tensor

// use both Tuuu and TIuu
void fast_tensor_power_method(CountSketch* cs_T, int dim, int rank, int L, int T, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double** v);
double fast_collide_tensor_power_method(CountSketch* cs_T, int dim, int rank, int L, int T, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double** v, bool report_residue = false);
// W: double[dim1][dim2]
// v: double[k][dim2]
void fast_kernel_tensor_power_method(CountSketch* cs_T, int dim1, int dim2, double* W, int rank, int L, int T, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, Matlab_wrapper* mat_wrapper, double* lambda, double** v);

void slow_tensor_power_method(Tensor* tensor, int dim, int rank, int L, int T, double* lambda, double** v);

void slow_kernel_tensor_power_method(Tensor* tensor, int dim1, int dim2, double* W, int rank, int L, int T, Matlab_wrapper* mat_wrapper, double* lambda, double** v);

// A: rank x dim
// Note:  in the following two processes, assume A has already been initialized
void slow_ALS(Tensor* tensor, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, double* lambda, double* A);
void fast_ALS(CountSketch* cs_T, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double* A);

void slow_asym_ALS(Tensor* tensor, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, double* lambda, double* A, double* B, double* C);
// If report_residue is set to true: return ||T-...||_F^2 / ||T||^2. might take a lot of time
double fast_asym_ALS(AsymCountSketch* cs_T, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double* A, double* B, double* C, bool report_residue = false);
double fast_asym_tensor_power_method(AsymCountSketch* cs_T, int dim, int rank, int L, int T, Matlab_wrapper*, FFT_wrapper*, FFT_wrapper*, double* lambda, double** v, bool report_residue = false);

//new methods created by zhao
double random01();
void generate_list_of_random_numbers(double *a, int m);
void generate_list_of_sorted_random_numbers(double *a, int m);
void assign_random_numbers_to_buckets(double *a, int m, double *q, int n, int *bucket);
void generate_random_permutation(int *c, int m);
void generate_cumulative_probablity(double *u, double *q, int n);
double fast_sampling_binary_search_Tuuu(Tensor* T, double *u, int n, int B, int b);
void   fast_sampling_binary_search_TIuu(Tensor* T, double *u, int n, double *u_out, int B, int b);

double fast_sampling_linear_merge_Tuuu(Tensor* T, double *u, int n, int B, int b, int rank, double *old_lambda, double **old_v);
void   fast_sampling_linear_merge_TIuu(Tensor* T, double *u, int n, double *u_out, int B, int b, int rank, double * old_lambda, double **old_v);

double fast_sampling_Tuuu(Tensor* T, double *u, int n, int B, int b, int rank, double * old_lambda, double **old_v);
void   fast_sampling_TIuu(Tensor* T, double* u, int n, double* u_out, int B, int b,int rank, double * old_lambda, double **old_v);

void fast_sampling_tensor_power_method(Tensor* T, int dim, int rank, int L, int iterT, int B,int b, double *lambda, double ** v );
void prescan_sampling_tensor_power_method(Tensor* T, int dim, int rank, int L, int iterT, int B,int b, int* slice_b, int max_b, double *lambda, double ** v );

/////////////////////////////////////
void normalize_vector(double *input_vector, double *output_vector, int dim);

double fast_sample_asym_ALS(Tensor* tensor, int dim, int rank, int para_T, int para_B, int para_b, Matlab_wrapper* mat_wrapper, double * lambda, double *A, double *B, double *C);

#endif
