#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

#include "fftw3.h"
#include "util.h"
#include "tensor.h"
#include "hash.h"
#include "count_sketch.h"
#include "fft_wrapper.h"
#include "matlab_wrapper.h"
#include "fast_tensor_power_method.h"
#define DEBUG 1

int para_B = 20;
int para_b = 11;
int K = 6;
int dim = 200;
int rank = 10;
double nsr = 0.01;
int nnz = 0;
bool diag_only = false;
double rate = 0.01;
int TR = 10;
int para_L, para_T;//L=50, T=30

//int para_m=10;//zhao's new parameter.

double decay = 0.8;
enum DECAY_METHOD {UNIFORM, INVERSE, INVERSE_SQUARE, LINEAR};
int decay_method = UNIFORM;

int dim1 = 100;
int dim2 = 100;

int evaldim = 1000;
double lnr_sigma = 2.0;

double thr = 0.1;

Tensor* T = new Tensor();
Hashes* hashes;
Hashes* asym_hashes[3];
FFT_wrapper *fft_wrapper, *ifft_wrapper;
Matlab_wrapper* mat_wrapper;

Tensor* gen_full_rank_tensor(size_t dim) {

	Tensor* ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE);
	for(int i = 0; i < dim; i++)
		for(int j = i; j < dim; j++)
			for(int k = j; k < dim; k++) {
				//double t = generate_std_normal();
				double t = (double)rand() / RAND_MAX;
				ret->A[IND3D(i,j,k,dim)] = t;
				ret->A[IND3D(i,k,j,dim)] = t;
				ret->A[IND3D(j,i,k,dim)] = t;
				ret->A[IND3D(j,k,i,dim)] = t;
				ret->A[IND3D(k,i,j,dim)] = t;
				ret->A[IND3D(k,j,i,dim)] = t;
			}

	double norm = 0;
	for(size_t p = 0; p < dim*dim*dim; p++)
		norm += SQR(ret->A[p]);
	double scale = 1.0 / sqrt(norm);
	for(size_t p = 0; p < dim*dim*dim; p++)
		ret->A[p] *= scale;

	assert(ret->symmetric_check());
	assert(safe_compare(ret->sqr_fnorm(), 1) == 0);

	return ret;

}

Tensor* gen_low_rank_tensor(size_t dim, int rank, double decay, double nsr, bool orthogonal = true, bool use_log_normal = false, int decay_method = 0, const char* filename = NULL) {

	Tensor* ret = NULL;
	if (dim > 1500) {
		// use mmap()
		ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE, filename);
	}
	else {
		ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE);
	}

	double** u = new double*[rank];
	double* lambda = new double[rank];
	double t = 1;
	printf("decay method: %d\n", decay_method);
	for(int k = 0; k < rank; k++) {
		switch(decay_method) {
			case INVERSE:
				lambda[k] = 1.0 / (k+1);
				break;
			case INVERSE_SQUARE:
				lambda[k] = 1.0 / ((k+1) * (k+1));
				break;
			case LINEAR:
				lambda[k] = (rank - k) / (double)rank;
				break;
			default: // UNIFORM
				lambda[k] = 1.0;
				break;
		}
		printf("lambda[%d] = %f\n", k, lambda[k]);
		// t /= decay
		t -= decay;
		u[k] = new double[dim];
		for(int i = 0; i < dim; i++)
			//u[i]= (double)rand() / RAND_MAX + i % 2 * 10;
			//u[k][i] = (double)rand() / RAND_MAX - 0.5;
			//u[k][i] = (double)rand() / RAND_MAX;
			if (!use_log_normal)
                generate_uniform_sphere_point(dim, u[k]);
            else {
                mat_wrapper->lognrnd(dim, 1, lnr_sigma, u[k]);
                double sum = 0;
                for(int i = 0; i < dim; i++)
                    sum += SQR(u[k][i]);
                double scale = 1.0 / sqrt(sum);
                for(int i = 0; i < dim; i++)
                    u[k][i] *= scale;
            }
	}
    if (orthogonal) {
        puts("Doing gram-schmidt process");
        gram_schmidt_process(rank, dim, u);
    }
	puts("Generating entries");
	for(size_t i = 0; i < dim; i++) {
		putchar('.');
		fflush(stdout);
		for(size_t j = 0; j < dim; j++)
			for(size_t k = 0; k < dim; k++)
				for(size_t r = 0; r < rank; r++)
					ret->A[IND3D(i,j,k,dim)] += lambda[r] * u[r][i] * u[r][j] * u[r][k];
	}
	putchar('\n');
	puts("Computing Fnorm...");
	double fnorm = sqrt(ret->sqr_fnorm());
	double scale = 1.0 / fnorm;
	puts("Normalizing");
	size_t counter = 0;
	for(size_t p = 0; p < dim*dim*dim; p++) {
		counter++;
		if (counter == dim * dim) {
			putchar('.');
			fflush(stdout);
			counter = 0;
		}
		ret->A[p] *= scale;
	}
	puts("\nDoing symmtric check...");
	assert(ret->symmetric_check());
	puts("Doing Fnorm check...");
	assert(safe_compare(ret->sqr_fnorm(), 1.0) == 0);

	// add noise
	if (nsr > 1e-5) {
		puts("Adding noise");
		double sigma = sqrt(nsr / (dim*dim*dim));
		for(size_t i = 0; i < dim; i++) {

			for(size_t j = i; j < dim; j++)
				for(size_t k = j; k < dim; k++) {
					ret->A[IND3D(i,j,k,dim)] += sigma * generate_std_normal();
					double t = ret->A[IND3D(i,j,k,dim)];
					ret->A[IND3D(i,k,j,dim)] = t;
					ret->A[IND3D(j,i,k,dim)] = t;
					ret->A[IND3D(j,k,i,dim)] = t;
					ret->A[IND3D(k,i,j,dim)] = t;
					ret->A[IND3D(k,j,i,dim)] = t;
				}
		putchar('.');
		fflush(stdout);
		}
	}
	puts("\nDoing symmtric check...");
	assert(ret->symmetric_check(true));

    FILE* fcore = fopen("tensor.core.dat", "wb");
    assert(fcore);
    for(int k = 0; k < rank; k++)
        fwrite(u[k], sizeof(double), dim, fcore);
    fclose(fcore);
    
    for(int k = 0; k < rank; k++)
        delete[] u[k];
	delete[] u;
	return ret;

}

// Generate a sparse tensor with only n_zeros entries.
// Currently we still store it in dense format
Tensor* gen_sparse_tensor(size_t dim, int n_zeros, bool diag_only) {
	Tensor* ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE);
	int i, j, k;
	for(int l = 0; l < n_zeros; l++) {
				i = rand() % dim;
				double t = (double)rand() / RAND_MAX;
				if (diag_only) {
					j = k = i;	
					printf("%d,%d,%d = %f\n", i, j, k, t);
				} else {
					j = rand() % dim;
					k = rand() % dim;
				}
				ret->A[IND3D(i,j,k,dim)] = t;
				ret->A[IND3D(i,k,j,dim)] = t;
				ret->A[IND3D(j,i,k,dim)] = t;
				ret->A[IND3D(j,k,i,dim)] = t;
				ret->A[IND3D(k,i,j,dim)] = t;
				ret->A[IND3D(k,j,i,dim)] = t;
	}

	double norm = 0;
	for(size_t p = 0; p < dim*dim*dim; p++)
		norm += SQR(ret->A[p]);
	double scale = 1.0 / sqrt(norm);
	if (diag_only) 
		printf("scale = %f\n", scale);
	for(size_t p = 0; p < dim*dim*dim; p++)
		ret->A[p] *= scale;

	assert(ret->symmetric_check());
	assert(safe_compare(ret->sqr_fnorm(), 1) == 0);

	return ret;
}

// decay <= 0: linear decay
// eigens: double dim x dim
Tensor* gen_high_rank_tensor(size_t dim, double decay, double nsr, double* eigens) {
    
	Tensor* ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE);

	for(int k = 0; k < dim; k++) {
        generate_uniform_sphere_point(dim, eigens + k * dim);
    }
	double t = 1;

    for(int k = 0; k < dim; k++) {
        t = (decay <= 0)? 1.0 / (k+1) : t * decay;
        double* u = eigens + k * dim;
        for(int i1 = 0; i1 < dim; i1++)
            for(int i2 = 0; i2 < dim; i2++)
                for(int i3 = 0; i3 < dim; i3++)
                    ret->A[IND3D(i1,i2,i3,dim)] += t * u[i1] * u[i2] * u[i3];
    }

	double fnorm = sqrt(ret->sqr_fnorm());
	double scale = 1.0 / fnorm;
	for(size_t p = 0; p < dim*dim*dim; p++)
		ret->A[p] *= scale;

	assert(ret->symmetric_check());
	assert(safe_compare(ret->sqr_fnorm(), 1.0) == 0);

	// add noise
	double sigma = sqrt(nsr / (dim*dim*dim));
	for(int i = 0; i < dim; i++)
		for(int j = i; j < dim; j++)
			for(int k = j; k < dim; k++) {
				ret->A[IND3D(i,j,k,dim)] += sigma * generate_std_normal();
				double t = ret->A[IND3D(i,j,k,dim)];
				ret->A[IND3D(i,k,j,dim)] = t;
				ret->A[IND3D(j,i,k,dim)] = t;
				ret->A[IND3D(j,k,i,dim)] = t;
				ret->A[IND3D(k,i,j,dim)] = t;
				ret->A[IND3D(k,j,i,dim)] = t;
			}

	assert(ret->symmetric_check());
	return ret;    
    
}

Tensor* gen_gaussian_tensor(int dim, double nsr) {
    
	Tensor* ret = new Tensor(dim, TENSOR_STORE_TYPE_DENSE);
	// add noise
	double sigma = sqrt(nsr / (dim*dim*dim));
	for(int i = 0; i < dim; i++)
		for(int j = i; j < dim; j++)
			for(int k = j; k < dim; k++) {
				double t = sigma * generate_std_normal();
				ret->A[IND3D(i,j,k,dim)] = t; 
				ret->A[IND3D(i,k,j,dim)] = t;
				ret->A[IND3D(j,i,k,dim)] = t;
				ret->A[IND3D(j,k,i,dim)] = t;
				ret->A[IND3D(k,i,j,dim)] = t;
				ret->A[IND3D(k,j,i,dim)] = t;
			}

	double fnorm = sqrt(ret->sqr_fnorm());
	printf("fnorm = %f\n", fnorm);
	// double scale = 1.0 / fnorm;
	// for(size_t p = 0; p < dim*dim*dim; p++)
	//	ret->A[p] *= scale;

	assert(ret->symmetric_check());
	// assert(safe_compare(ret->sqr_fnorm(), 1.0) == 0);

	return ret;    
    
}

void task1() {
	system("/bin/mkdir -p ./data");
	char s[100];
	sprintf(s,"./data/tensor_dim_%d_rank_%d_noise_%.2lf_decaymethod_%d.dat",dim,rank,nsr,decay_method);
	T = gen_low_rank_tensor(dim, rank, 0, nsr, true, false, decay_method, s);
	T->save(s);

}

void task2() {
    
    double* eigens = new double[rank * rank];
    T = gen_high_rank_tensor(dim, -1, nsr, eigens);
    T->save("tensor.dat");
    
}

void task9() {
	system("/bin/mkdir -p ./data");
	char s[100];
	sprintf(s,"./data/tensor_dim_%d_nnz_%d_%s.dat",dim,nnz,diag_only ? "diag" : "random");
	T = gen_sparse_tensor(dim, nnz, diag_only);
	T->save(s);
}

void task10() {
	system("/bin/mkdir -p ./data");
	char s[100];
	sprintf(s,"./data/tensor_dim_%d_gaussian_%f.dat",dim,nsr);
	T = gen_gaussian_tensor(dim, nsr);
	T->save(s);
}

void task3(char* file1, char* file2) {
    // slow robust tensor power method
    
    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    printf("# [STAT]: fnorm=%lf\n", (T->sqr_fnorm()));
    
    double* lambda = new double[rank];
    double** u = new double*[rank];
    for(int k = 0; k < rank; k++)
        u[k] = new double[dim];

    clock_t start, stop;	//add by zhao
    struct timespec wall_start, wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    slow_tensor_power_method(T, dim, rank, para_L, para_T, lambda, u);

    stop = clock();		//add by zhao
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    long long wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: wall_time=%lf\n", wall_time / 1000000000.0);
    
    printf("# [STAT]: residue=%lf\n", (T->sqr_fnorm()));
    
    FILE* f = fopen(file2, "wb");
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&rank, sizeof(int), 1, f);
    fwrite(lambda, sizeof(double), rank, f);
    for(int k = 0; k < rank; k++)
        fwrite(u[k], sizeof(double), dim, f);
    fclose(f);
    
}

void task4(char* file1, char* file2) {
    
    // fast sketch-based robust tensor power method
    clock_t start, stop;	//add by zhao
    struct timespec wall_start, wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);

    stop = clock();		//add by zhao
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    long long wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("CPU time for loading tensor from disk to memory : %lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("Wall time for loading tensor from disk to memory : %lf\n", wall_time / 1000000000.0);
    
    T->symmetric_check();
    dim = T->dim;
    printf("squared T->fnorm = %lf\n", (T->sqr_fnorm()));
	//printf("first n entries of original Tensor:\n");
    	//for(int i=0; i<dim; i++)
    	//	printf("%lf, ", T->A[i]);
    	//printf("\n");
    
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    fft_wrapper = new FFT_wrapper(POWER2(para_b), FFTW_FORWARD);
    ifft_wrapper = new FFT_wrapper(POWER2(para_b), FFTW_BACKWARD);
    
    double* lambda = new double[rank];
    double** u = new double*[rank];
    for(int k = 0; k < rank; k++)
        u[k] = new double[dim];
    
    hashes = new Hashes(para_B, para_b, dim, K);
    CountSketch* cs_T = new CountSketch(hashes);
    cs_T->set_tensor(T);

    stop = clock();
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: prep_cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: prep_wall_time=%lf\n", wall_time / 1000000000.0);
    
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();
    
    fast_tensor_power_method(cs_T, dim, rank, para_L, para_T, fft_wrapper, ifft_wrapper, lambda, u);
    
    stop = clock();
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: wall_time=%lf\n", wall_time / 1000000000.0);
    
    double sum = 0;
    for(int i1 = 0; i1 < dim; i1 ++)
        for(int i2 = 0; i2 < dim; i2 ++) 
            for(int i3 = 0; i3 < dim; i3 ++) {
                double t = T->A[IND3D(i1, i2, i3, dim)];
                for(int k = 0; k < rank; k++)
                    t -= lambda[k] * u[k][i1] * u[k][i2] * u[k][i3];
                sum += SQR(t);
            }
  
    printf("# [STAT]: residue=%lf\n", (sum));
    printf("# [STAT]: fnorm=%lf\n", (T->sqr_fnorm()));
    
    FILE* f = fopen(file2, "wb");
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&rank, sizeof(int), 1, f);
    fwrite(lambda, sizeof(double), rank, f);
    for(int k = 0; k < rank; k++)
        fwrite(u[k], sizeof(double), dim, f);
    fclose(f);    
    
}

void evaluate_and_save(Tensor* T, int dim, int rank, double* lambda, double* A, double* B, double* C, char* filename) {
    
    double sum = 0;
    for(int i1 = 0; i1 < dim; i1 ++)
        for(int i2 = 0; i2 < dim; i2 ++)
            for(int i3 = 0; i3 < dim; i3 ++) {
                double t = T->A[IND3D(i1, i2, i3, dim)];
                for(int k = 0; k < rank; k++)
                    t -= lambda[k] * A[IND2D(k, i1, dim)] * B[IND2D(k, i2, dim)] * C[IND2D(k, i3, dim)];
                sum += SQR(t);
            }
    printf("# [STAT]: residue=%lf\n", (sum));
    printf("# [STAT]: fnorm=%lf\n", (T->sqr_fnorm()));
    
    FILE* f = fopen(filename, "wb");
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&rank, sizeof(int), 1, f);
    fwrite(lambda, sizeof(double), rank, f);
    for(int k = 0; k < rank; k++)
        fwrite(A + k * dim, sizeof(double), dim, f);
    for(int k = 0; k < rank; k++)
        fwrite(B + k * dim, sizeof(double), dim, f);
    for(int k = 0; k < rank; k++)
        fwrite(C + k * dim, sizeof(double), dim, f);    
    fclose(f);     
    
}

void task5(char* file1, char* file2) {
    
    // slow ALS
    
    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    printf("squared T->fnorm = %lf\n", (T->sqr_fnorm()));

    double* U = new double[dim * rank];
    double* UT = new double[rank * dim];
    mat_wrapper->svds(T->A, dim, rank, U, NULL, NULL);    
    for(int k = 0; k < rank; k++)
        for(int i = 0; i < dim; i++)
            UT[IND2D(k,i,dim)] = U[IND2D(i,k,rank)];
    
    double* lambda = new double[rank];
    double* A = new double[rank * dim];
    double* B = new double[rank * dim];
    double* C = new double[rank * dim];
    for(int k = 0; k < rank; k++) {
        memcpy(A + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(B + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(C + k * dim, UT + k * dim, sizeof(double) * dim);
    }

    clock_t start, stop;	//add by zhao
    struct timespec wall_start, wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    slow_asym_ALS(T, dim, rank, para_T, mat_wrapper, lambda, A, B, C);

    stop = clock();		//add by zhao
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    long long wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: wall_time=%lf\n", wall_time / 1000000000.0);
    
    evaluate_and_save(T, dim, rank, lambda, A, B, C, file2);
    
}

void task6(char* file1, char* file2) {
    
    // fast ALS
    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    printf("squared T->fnorm = %lf\n", (T->sqr_fnorm()));

    clock_t start, stop;	//add by zhao
    struct timespec wall_start, wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    double* U = new double[dim * rank];
    double* UT = new double[rank * dim];
    mat_wrapper->svds(T->A, dim, rank, U, NULL, NULL); 
    for(int k = 0; k < rank; k++)
        for(int i = 0; i < dim; i++)
            UT[IND2D(k,i,dim)] = U[IND2D(i,k,rank)];
    
    double* lambda = new double[rank];
    double* AA = new double[rank * dim];
    double* BB = new double[rank * dim];
    double* CC = new double[rank * dim];
    
    for(int k = 0; k < rank; k++) {
        memcpy(AA + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(BB + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(CC + k * dim, UT + k * dim, sizeof(double) * dim);
    }
    
    fft_wrapper = new FFT_wrapper(POWER2(para_b), FFTW_FORWARD);
    ifft_wrapper = new FFT_wrapper(POWER2(para_b), FFTW_BACKWARD);
    for(int i = 0; i < 3; i++) {
        asym_hashes[i] = new Hashes(para_B, para_b, dim, 6);
        asym_hashes[i]->to_asymmetric_hash();
    }
    AsymCountSketch* cs_T = new AsymCountSketch(3, asym_hashes);
    cs_T->set_tensor(T);

    stop = clock();		//add by zhao
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    long long wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: prep_cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: prep_wall_time=%lf\n", wall_time / 1000000000.0);

    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    fast_asym_ALS(cs_T, dim, rank, para_T, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, AA, BB, CC);

    stop = clock();		//add by zhao
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: wall_time=%lf\n", wall_time / 1000000000.0);
    evaluate_and_save(T, dim, rank, lambda, AA, BB, CC, file2);
    
}



void task7(char* file1, char* file2) {
    
    // fast sampling-based robust tensor power method
    
    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    // use memory mapped file
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    
	double* lambda = new double[rank];
	double** u = new double*[rank];
	for(int k = 0; k < rank; k++)
		u[k] = new double[dim];

	clock_t start, stop;	//add by zhao
	struct timespec wall_start, wall_end;
	clock_gettime(CLOCK_MONOTONIC, &wall_start);
	start = clock();

	fast_sampling_tensor_power_method(T, dim, rank, para_L, para_T, para_B, para_b, lambda, u);

	stop = clock();		//add by zhao
	clock_gettime(CLOCK_MONOTONIC, &wall_end);
	long long wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
	printf("# [STAT]: cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
	printf("# [STAT]: wall_time=%lf\n", wall_time / 1000000000.0);

    double sum = 0;
    for(int i1 = 0; i1 < dim; i1 ++)
        for(int i2 = 0; i2 < dim; i2 ++) 
            for(int i3 = 0; i3 < dim; i3 ++) {
                double t = T->A[IND3D(i1, i2, i3, dim)];
                for(int k = 0; k < rank; k++)
                    t -= lambda[k] * u[k][i1] * u[k][i2] * u[k][i3];
                sum += SQR(t);
            }
    printf("# [STAT]: residue=%lf\n", (sum));
    printf("# [STAT]: fnorm=%lf\n", (T->sqr_fnorm()));

    FILE* f = fopen(file2, "wb");
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&rank, sizeof(int), 1, f);
    fwrite(lambda, sizeof(double), rank, f);
    for(int k = 0; k < rank; k++)
        fwrite(u[k], sizeof(double), dim, f);
    fclose(f);    
    
}


void task8(char* file1, char* file2) {
    
    // fast sampling ALS 

    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;
    printf("squared T->fnorm = %lf\n", (T->sqr_fnorm()));

    double* U = new double[dim * rank];
    double* UT = new double[rank * dim];
    mat_wrapper->svds(T->A, dim, rank, U, NULL, NULL);    
    for(int k = 0; k < rank; k++)
        for(int i = 0; i < dim; i++)
            UT[IND2D(k,i,dim)] = U[IND2D(i,k,rank)];
    
    double* lambda = new double[rank];
    double* A = new double[rank * dim];
    double* B = new double[rank * dim];
    double* C = new double[rank * dim];
    for(int k = 0; k < rank; k++) {
        memcpy(A + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(B + k * dim, UT + k * dim, sizeof(double) * dim);
        memcpy(C + k * dim, UT + k * dim, sizeof(double) * dim);
    }

    clock_t start, stop;	//add by zhao
    struct timespec wall_start, wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    fast_sample_asym_ALS(T, dim, rank, para_T, para_B, para_b, mat_wrapper, lambda, A, B, C);

    stop = clock();		//add by zhao
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    long long wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: wall_time=%lf\n", wall_time / 1000000000.0);
    
    evaluate_and_save(T, dim, rank, lambda, A, B, C, file2);

    
}

void task11(char* file1, char* file2) {
    
    // fast sampling-based robust tensor power method with pre-scanning

    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    // use memory mapped file
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;

    double* lambda = new double[rank];
    double** u = new double*[rank];
    for(int k = 0; k < rank; k++)
	u[k] = new double[dim];

    clock_t start, stop;	//add by zhao
    struct timespec wall_start, wall_end;
    
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    double *slice_fnorms = new double[dim];
    int *slice_b = new int[dim];
    int max_b = 0;
    T->sqr_slice_fnorms(slice_fnorms);
    double total_fnorm = 0.0;
    for(int i = 0; i < dim; ++i) {
	    total_fnorm += slice_fnorms[i];
    }
    for(int i = 0; i < dim; ++i) {
	    slice_b[i] = dim * para_b * slice_fnorms[i] / total_fnorm + 0.5;
	    printf("Slice %d has %d samples\n", i, slice_b[i]);
	    if (max_b < slice_b[i])
		    max_b = slice_b[i];
    }
    printf("Max b = %d\n", max_b);

    stop = clock();		//add by zhao
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    long long wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: prep_cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: prep_wall_time=%lf\n", wall_time / 1000000000.0);
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    start = clock();

    prescan_sampling_tensor_power_method(T, dim, rank, para_L, para_T, para_B, para_b, slice_b, max_b, lambda, u);

    stop = clock();		//add by zhao
    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    wall_time = 1000000000L * (wall_end.tv_sec - wall_start.tv_sec) + wall_end.tv_nsec - wall_start.tv_nsec;
    printf("# [STAT]: cpu_time=%lf\n", (stop-start)/ (1.0*CLOCKS_PER_SEC));
    printf("# [STAT]: wall_time=%lf\n", wall_time / 1000000000.0);

    double sum = 0;
    for(int i1 = 0; i1 < dim; i1 ++)
	for(int i2 = 0; i2 < dim; i2 ++) 
	    for(int i3 = 0; i3 < dim; i3 ++) {
		double t = T->A[IND3D(i1, i2, i3, dim)];
		for(int k = 0; k < rank; k++)
		    t -= lambda[k] * u[k][i1] * u[k][i2] * u[k][i3];
		sum += SQR(t);
	    }
    printf("# [STAT]: residue=%lf\n", (sum));
    printf("# [STAT]: fnorm=%lf\n", (T->sqr_fnorm()));

    FILE* f = fopen(file2, "wb");
    fwrite(&dim, sizeof(int), 1, f);
    fwrite(&rank, sizeof(int), 1, f);
    fwrite(lambda, sizeof(double), rank, f);
    for(int k = 0; k < rank; k++)
	fwrite(u[k], sizeof(double), dim, f);
    fclose(f);    
    delete [] slice_fnorms;
    
}


void task12(char* file1) {
    
    // collect tensor statistics

    Tensor* T = new Tensor(1, TENSOR_STORE_TYPE_DENSE);
    // use memory mapped file
    T->load(file1, TENSOR_STORE_TYPE_DENSE);
    T->symmetric_check();
    dim = T->dim;

    
    double *slice_fnorms = new double[dim];
    double *slice_mean = new double[dim];
    double *slice_variance = new double[dim];
    int *slice_b = new int[dim];
    
    T->sqr_slice_fnorms(slice_fnorms);
    T->slice_stats(slice_mean, slice_variance);
    double total_fnorm = 0.0;
    for(int i = 0; i < dim; ++i) {
	    total_fnorm += slice_fnorms[i];
    }
    for(int i = 0; i < dim; ++i) {
	    slice_b[i] = dim * para_b * slice_fnorms[i] / total_fnorm + 0.5;
    }

    for (int i = 0; i < dim; ++i) {
        printf("slice = %4d, fnorm^2 = %12.6g, b = %5d, mean = %12.6g, variance = %12.6g\n", i, slice_fnorms[i], slice_b[i], slice_mean[i], slice_variance[i]);
    }

    delete[] slice_fnorms;
    delete[] slice_mean;
    delete[] slice_variance;

}

int main(int argc, char* argv[]) {

	srand(time(0));
    
    mat_wrapper = new Matlab_wrapper();

	assert(argc >= 2);

	if (strcmp(argv[1], "synth_lowrank") == 0) {
        if (argc < 5) {
		printf("Usage: %s synth_lowrank dim rank noise decay\n", argv[0]);
		exit(0);
	}
        dim = atoi(argv[2]);
        rank = atoi(argv[3]);
        nsr = atof(argv[4]);
	
	if (argc >= 6)
		decay_method = atoi(argv[5]);
        
	task1();
        
	}
    else if (strcmp(argv[1], "synth_highrank") == 0) {
        
        dim = atoi(argv[2]);
        nsr = atof(argv[3]);
        
        task2();
    }
    else if (strcmp(argv[1], "synth_sparse") == 0) {
	dim = atoi(argv[2]);
	nnz = atoi(argv[3]);
	if (argc >= 5)
		diag_only = atoi(argv[4]);
	task9();
    }
    else if (strcmp(argv[1], "synth_gaussian") == 0) {
	dim = atoi(argv[2]);
	nsr = atof(argv[3]);
	task10();
    }
    else if (strcmp(argv[1], "slow_rbp") == 0) {
        
        rank = atoi(argv[3]);
        para_L = atoi(argv[4]);
        para_T = atoi(argv[5]);
        
        task3(argv[2], argv[6]);
        
    }
    else if (strcmp(argv[1], "fast_rbp") == 0) {
        
        rank = atoi(argv[3]);
        para_L = atoi(argv[4]);
        para_T = atoi(argv[5]);
        para_B = atoi(argv[6]);
        para_b = atoi(argv[7]);
        
        printf("fast_rbp algorithm set: \n");
	printf("rank = %d, para_L = %d, para_T = %d, B = %d, b = %d\n", rank, para_L, para_T, para_B, para_b);
        task4(argv[2], argv[8]);
        
    }
    else if (strcmp(argv[1], "slow_als") == 0) {
        
        rank = atoi(argv[3]);
        para_T = atoi(argv[4]);
        task5(argv[2], argv[5]);
        
    }
    else if (strcmp(argv[1], "fast_als") == 0) {
        
        rank = atoi(argv[3]);
        para_T = atoi(argv[4]);
        para_B = atoi(argv[5]);
        para_b = atoi(argv[6]);

	printf("fast_als algorithm set: \n");
	printf("rank = %d, para_T = %d, B = %d, b = %d\n", rank, para_T, para_B, para_b);
        
        task6(argv[2], argv[7]);
        
    }
    else if(strcmp(argv[1], "fast_sample_rbp") == 0) {
	rank = atoi(argv[3]);
        para_L = atoi(argv[4]);
        para_T = atoi(argv[5]);
        para_B = atoi(argv[6]); // number of samples
        para_b = atoi(argv[7]);
        
        printf("fast_sample_rbp algorithm set : \n");
        printf("rank = %d, L = %d, T = %d, B = %d, b = %d\n",rank,para_L,para_T,para_B,para_b);
        
        task7(argv[2], argv[8]);
        
    }
    else if(strcmp(argv[1], "prescan_sample_rbp") == 0) {
	rank = atoi(argv[3]);
        para_L = atoi(argv[4]);
        para_T = atoi(argv[5]);
        para_B = atoi(argv[6]); // number of samples
        para_b = atoi(argv[7]);
        
        printf("prescan_sample_rbp algorithm set : \n");
        printf("rank = %d, L = %d, T = %d, B = %d, b = %d\n",rank,para_L,para_T,para_B,para_b);
        if (rank > 1) {
		printf("RANK > 1 unimplemented!\n");
		return 1;
	}
        task11(argv[2], argv[8]);
        
    }
   else if (strcmp(argv[1], "fast_sample_als") == 0) {
        
        rank = atoi(argv[3]);
        para_T = atoi(argv[4]);
        para_B = atoi(argv[5]);
        para_b = atoi(argv[6]);

        printf("fast_sample_als algorithm set : \n");
        printf("rank = %d, T = %d, B= %d, b = %d\n", rank, para_T, para_B, para_b);
        
        task8(argv[2], argv[7]);
        
    }
    else if (strcmp(argv[1], "tensor_stat") == 0) {
        if (argc > 3) 
		para_b = atoi(argv[3]);
	else
		para_b = 20;
        task12(argv[2]);
    }

	return 0;

}
