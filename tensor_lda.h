#ifndef TENSOR_LDA_H_
#define TENSOR_LDA_H_

#include "hash.h"
#include "corpus.h"
#include "LDA.h"
#include "matlab_wrapper.h"
#include "fft_wrapper.h"
#include "count_sketch.h"

// M1 (returned): double V
void compute_word_frequency(Corpus* corpus, double* M1);

// M1: double V, 
// W (returned): double V x K
// A2WW (returned): double K x K
void slow_whiten(Corpus* corpus, double alpha0, double* M1, double* W, double* A2WW, Matlab_wrapper*);
void multithread_slow_whiten(int n_threads, Corpus* corpus, double alpha0, double* M1, double* W, Matlab_wrapper*);

void tensor_lda_parameter_recovery(LDA* model, double* W, size_t V, int K, double alpha0, double* lambda, double** v, Matlab_wrapper* mat_wrapper);

// W: double V x K
// model (returned): fill in alpha and Phi
// T: # of random initializations; set to K by default
// L: # of iterations per initialization; set to 30 by default
// slow_tensor_lda: store M3(W, W, W) explicityly and invoke slow_tensor_power_method
void slow_tensor_lda(Corpus* corpus, double alpha0, double* W, double* A2WW, LDA* model, Matlab_wrapper*, int T = 30, int L = 30);

AsymCountSketch* fast_tensor_lda(Corpus* corpus, double alpha0, int B, int b, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper);
double slow_tensor_kernel_eval_tuuu(Corpus* corpus, double alpha0, double* u);
double fast_tensor_kernel_eval_tuuu(Corpus* corpus, double alpha0, double* u);

void slow_als_lda(Corpus* corpus, double alpha0, double* W, LDA* model, int B, int b, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, int T = 30);

// method = 0: ALS; method = 1: robust tensor power method
void fast_whiten_tensor_lda(int method, Corpus* corpus, double alpha0, double* W, LDA* model, int B, int b, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, int T = 20, int L = 30);
void fast_symmetric_whiten_tensor_lda(int method, Corpus* corpus, double alpha0, double* W, LDA* model, int B, int b, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, int T = 20, int L = 30);

// if use_brute_force is set to true, then we use brute force method for tensor decomposition: valid for low $k$ values
void multithread_fast_whiten_tensor_lda(int method, bool use_brute_force, int n_threads, Corpus* corpus, double alpha0, double* W, LDA* model, int B, int b, Matlab_wrapper* mat_wrapper, int T = 20, int L = 30, double* times = NULL);

#endif
