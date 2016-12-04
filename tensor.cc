#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>

#include "config.h"
#include "util.h"
#include "tensor.h"
#include "matlab_wrapper.h"

Tensor::Tensor() {
	store_type = TENSOR_STORE_TYPE_NULL;
	A = NULL;
	dim = 0;
	values = NULL;
	idx[0] = NULL;
	idx[1] = NULL;
	idx[2] = NULL;	
	Lambda = NULL;
	U = NULL;
	rank = 0;
	is_mmapped = false;
	fd = 0;
}

Tensor::Tensor(size_t dim, int store_type, const char * mmapped_file) {
	if (store_type == TENSOR_STORE_TYPE_DENSE) {
		this->store_type = store_type;
		this->dim = dim;
		if (mmapped_file == NULL) {
			assert(dim > 0 && dim <= MAX_DENSE_TENSOR_DIM);
			A = new double[dim*dim*dim];
			memset(A, 0, sizeof(double) * dim*dim*dim);
		}
		else {
			// create a memory mapped file!
			is_mmapped = true;
			mapping_size = sizeof(int) + sizeof(double) * (size_t) dim * (size_t) dim * (size_t) dim;
			int mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
			fd = open(mmapped_file, O_RDWR | O_CREAT | O_TRUNC, mode);
			assert(fd);
			ftruncate(fd, mapping_size);
			printf("Mapping tensor size %lu\n", mapping_size);
			memblock = (char *)mmap(NULL, mapping_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			if (memblock == MAP_FAILED) {
				printf("mmap() failure!");
				perror("mmap");
				putchar('\n');
				exit(-1);
			}
			*(int*)memblock = dim;
			A = (double*)(memblock + sizeof(int));
		}
		values = NULL;
		idx[0] = NULL;
		idx[1] = NULL;
		idx[2] = NULL;
	
		this->Lambda = NULL;
		this->U = NULL;
		this->rank = 0;        

	}
	else if (store_type == TENSOR_STORE_TYPE_SPARSE) {

		this->A = NULL;

		this->store_type = store_type;
		this->dim = dim;
		this->nnz_count = 0;
		this->rate = 1;
        
		values = NULL;
		idx[0] = NULL;
		idx[1] = NULL;
		idx[2] = NULL;        
        
        this->Lambda = NULL;
        this->U = NULL;
        this->rank = 0;        

	}
	else {
		assert(0);
	}

}

Tensor::Tensor(size_t dim, int rank, int store_type) {

    if (store_type == TENSOR_STORE_TYPE_LOW_RANK) {
    
        this->store_type = TENSOR_STORE_TYPE_LOW_RANK;
        this->dim = dim;
        this->rank = rank;
        
        values = NULL;
		idx[0] = NULL;
		idx[1] = NULL;
		idx[2] = NULL;  
    
        A = new double[dim*dim];
        assert(A);
        memset(A, 0, sizeof(double) * dim*dim);
        
        Lambda = new double[rank];
        assert(Lambda);
        memset(Lambda, 0, sizeof(double) * rank);
        U = new double[rank * dim];
        memset(U, 0, sizeof(double) * rank * dim);
    
    }
    else {
        assert(0);
    }

}

Tensor::~Tensor() {
	clear();
}

int Tensor::symmetric_check(bool forced) {

	if (store_type == TENSOR_STORE_TYPE_DENSE) {
		if ((dim > MAX_DENSE_TENSOR_DIM && is_mmapped) && !forced) {
			printf("symmetric check skipped due to large dimension.\n");
			return 1;
		}
		if (getenv("TENSOR_BENCHMARK") != NULL) {
			puts("symmtric check disabled in benchmarking mode");
			return 1;
		}

		for(size_t i = 0; i < dim; i++)
			for(size_t j = i; j < dim; j++)
				for(size_t k = i; k < dim; k++) {
					double t = A[IND3D(i,j,k,dim)];
					if (safe_compare(A[IND3D(i,j,k,dim)], t) ||
						safe_compare(A[IND3D(i,k,j,dim)], t) ||
						safe_compare(A[IND3D(j,i,k,dim)], t) ||
						safe_compare(A[IND3D(j,k,i,dim)], t) ||
						safe_compare(A[IND3D(k,i,j,dim)], t) ||
						safe_compare(A[IND3D(k,j,i,dim)], t)) {
							return 0;
					}
				}

	}
	else if (store_type == TENSOR_STORE_TYPE_SPARSE) {

		for(int i = 0; i < nnz_count; i++)
			if (!(idx[0][i] <= idx[1][i] && idx[1][i] <= idx[2][i]))
				return 0;
		return 1;

	}
    else if (store_type == TENSOR_STORE_TYPE_LOW_RANK) {
        
        return 1;
        
    }
	else {
		assert(0);
	}

	return 1;

}

void Tensor::load(char* filename, int store_type, bool memory_mapped) {

	if (store_type == TENSOR_STORE_TYPE_DENSE) {

		clear();
		this->store_type = TENSOR_STORE_TYPE_DENSE;
		// get file size
		fd = open(filename, O_RDONLY);
		assert(fd);
		struct stat sb;
		fstat(fd, &sb);
		// for large tensor, we need to use mmap
		if (sb.st_size > 68719476736LL)
			memory_mapped = true;
		if (memory_mapped) {
			is_mmapped = true;
			printf("Mapping tensor size %lu\n", (uint64_t)sb.st_size);
			memblock = (char *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
			if (memblock == MAP_FAILED) {
				printf("mmap() failure!");
				perror("mmap");
				putchar('\n');
				exit(-1);
			}
			dim = *(int*)memblock;
			A = (double*)(memblock + sizeof(int));
		}
		else {
			close(fd);
			FILE* fp = fopen(filename, "rb");
			assert(fp);
			fread(&dim, sizeof(int), 1, fp);
			assert(dim > 0 && dim <= MAX_DENSE_TENSOR_DIM);

			A = new double[dim*dim*dim];
			for(int i = 0; i < dim; i++) {
				fread(A + i * dim * dim, sizeof(double), dim * dim, fp);
			}

			fclose(fp);
		}

		assert(symmetric_check());

	}
    else if (store_type == TENSOR_STORE_TYPE_LOW_RANK) {
    
        clear();
        this->store_type = TENSOR_STORE_TYPE_LOW_RANK;
        
        FILE* fp = fopen(filename, "rb");
        assert(fp);
        
        fread(&dim, sizeof(int), 1, fp);
        assert(dim >= 1);
        fread(&rank, sizeof(int), 1, fp);
        assert(1 <= rank && rank <= dim && rank <= MAX_DENSE_TENSOR_DIM);
        
        A = new double[dim*dim];
        memset(A, 0, sizeof(double) * dim*dim);
        Lambda = new double[rank];
        U = new double[rank * dim];
        
        fread(Lambda, sizeof(double), rank, fp);
        for(int k = 0; k < rank; k++)
            fread(U + k * dim, sizeof(double), dim, fp);
        
        fclose(fp);
    
    }
	else {
		assert(0);
	}

}


void Tensor::save(char* filename) {

	if (store_type == TENSOR_STORE_TYPE_DENSE) {
		if (!is_mmapped) {
			FILE* fp = fopen(filename, "wb");
			assert(fp);

			fwrite(&dim, sizeof(int), 1, fp);
			for(int i = 0; i < dim; i++) {
				fwrite(A + i * dim * dim, sizeof(double), dim * dim, fp);
			}

			fclose(fp);
		}
		else {
			if(msync(memblock, mapping_size, MS_SYNC)) {
				printf("msync() error\n");
				perror("msync");
			}
		}
	}
    else if (store_type == TENSOR_STORE_TYPE_LOW_RANK) {
    
        FILE* fp = fopen(filename, "wb");
        assert(fp);
        
        fwrite(&dim, sizeof(int), 1, fp);
        fwrite(&rank, sizeof(int), 1, fp);
        fwrite(Lambda, sizeof(double), rank, fp);
        for(int k = 0; k < rank; k++) {
            fwrite(U + k * dim, sizeof(double), dim, fp);
        }
        
        fclose(fp);
    
    }
	else {
		assert(0);
	}

}

void Tensor::load_view(int p, Matlab_wrapper* mat_wrapper) {

    double* LambdaU = new double[rank];
    assert(LambdaU);

    Engine* ep = mat_wrapper->get_engine();
    mxArray* mxU = mat_wrapper->create_matrix(U, rank, dim, false);
    assert(mxU);
    mxArray* mxLambda = mxCreateDoubleMatrix(1, rank, mxREAL);
    assert(mxLambda);
    if (p < 0)
        memcpy(LambdaU, Lambda, sizeof(double) * rank);
    else {
        for(int k = 0; k < rank; k++)
            LambdaU[k] = Lambda[k] * U[k * dim + p];
    }
    memcpy(mxGetPr(mxLambda), LambdaU, sizeof(double) * rank);    
    
    engPutVariable(ep, "U", mxU);
    engPutVariable(ep, "Lambda", mxLambda);
    engEvalString(ep, "A = U' * diag(Lambda) * U;");
    mxArray* mxA = engGetVariable(ep, "A");
    engEvalString(ep, "clear");
    
    memcpy(A, mxGetPr(mxA), sizeof(double) * dim * dim);
    
    mxDestroyArray(mxU);
    mxDestroyArray(mxLambda);
    mxDestroyArray(mxA);
    delete[] LambdaU;

}

// W: dim x rank
Tensor* Tensor::whiten(Matlab_wrapper* mat_wrapper, double *W) {

    assert(store_type == TENSOR_STORE_TYPE_LOW_RANK);
    this->load_view(-1, mat_wrapper);
    
    double* S = new double[rank];
    mat_wrapper->svds(A, dim, rank, W, S, NULL);
    printf("sigma(%d) = %lf\n", rank, S[rank-1]);
    for(int k = 0; k < rank; k++)
        S[k] = 1.0 / sqrt(S[k]);
    for(int i = 0; i < dim; i++)
        for(int k = 0; k < rank; k++)
            W[IND2D(i,k,rank)] *= S[k];
    delete[] S;
            
    Tensor* ret = new Tensor(rank, TENSOR_STORE_TYPE_DENSE);
    Tensor* t = new Tensor(rank, rank, TENSOR_STORE_TYPE_LOW_RANK);
    memcpy(t->Lambda, this->Lambda, sizeof(double) * rank);
    mat_wrapper->multiply(U, W, rank, dim, dim, rank, rank, rank, false, false, t->U);
    
    for(int i = 0;  i < rank; i++) {
        t->load_view(i, mat_wrapper);
        memcpy(ret->A + i * rank * rank, t->A, sizeof(double) * rank * rank);
    }
    
    delete t;
    return ret;
            
}

double Tensor::Tuuu(double* u, bool symeval) {

	double ret = 0;

	if (store_type == TENSOR_STORE_TYPE_DENSE) {

		size_t p = 0;
		if (!symeval) {
		for(size_t i = 0; i < dim; i++)
			for(size_t j = 0; j < dim; j++)
				for(size_t k = 0; k < dim; k++)
					ret += A[p++] * u[i] * u[j] * u[k];
		}
		else {
		    for(size_t i = 0; i < dim; i++)
		        for(size_t j = i; j < dim; j++)
		            for(size_t k = j; k < dim; k++) {
		                int cnt;
		                if (i == j && j == k) cnt = 1;
		                else if (i == j || j == k || i == k) cnt = 3;
		                else cnt = 6;
		                ret += cnt * A[IND3D(i,j,k,dim)] * u[i] * u[j] * u[k];
		            }
		}

	}
	else if (store_type == TENSOR_STORE_TYPE_SPARSE) {

		for(int p = 0; p < nnz_count; p++) {

			int i = idx[0][p], j = idx[1][p], k = idx[2][p];
			int t = (int)(i != j) + (int)(j != k); // 0, 1 or 2
			int mult = t + (1 << t); // 0 --> 1, 1 --> 3, 2 --> 6
			ret += mult * values[p] * u[i] * u[j] * u[k];

		}

	}
	else {
		assert(0);
	}

	return ret;

}

void Tensor::TIuu(double* u, double* ret, bool symeval) {

	memset(ret, 0, sizeof(double) * dim);

	if (store_type == TENSOR_STORE_TYPE_DENSE) {

		size_t p = 0;
		if (!symeval) {
		for(size_t i = 0; i < dim; i++)
			for(size_t j = 0; j < dim; j++)
				for(size_t k = 0; k < dim; k++)
					ret[i] += A[p++] * u[j] * u[k];
	    }
	    else {
	        for(int i = 0; i < dim; i++) {
	            for(int j = 0; j < dim; j++)
	                for(int k = j; k < dim; k++) {
	                    int cnt = (j == k)? 1 : 2;
	                    ret[i] += cnt * A[IND3D(i,j,k,dim)] * u[j] * u[k];
	                }
	        }
	    }

	}
	else if (store_type == TENSOR_STORE_TYPE_SPARSE) {

		for(int p = 0; p < nnz_count; p++) {

			int i = idx[0][p], j = idx[1][p], k = idx[2][p];

			if (i == j && j == k) {
				ret[i] += values[p] * u[i] * u[i];
			}
			else if (i == j) {
				ret[i] += 2 * values[p] * u[j] * u[k];
				ret[k] += values[p] * u[i] * u[i];
			}
			else if (j == k) {
				ret[k] += 2 * values[p] * u[i] * u[k];
				ret[i] += 2 * values[p] * u[k] * u[k];
			}
			else {
				ret[i] += 2 * values[p] * u[j] * u[k];
				ret[j] += 2 * values[p] * u[i] * u[k];
				ret[k] += 2 * values[p] * u[i] * u[j];
			}

		}

	}
	else {
		assert(0);
	}

}

void Tensor::TIuv(double* u, double* v, double* ret) {

    if (store_type == TENSOR_STORE_TYPE_DENSE) {
    
        memset(ret, 0, sizeof(double) * dim);
        for(size_t i = 0; i < dim; i++)
            for(size_t j = 0; j < dim; j++)
                for(size_t k = 0; k < dim; k++)
                    ret[i] += A[IND3D(i,j,k,dim)] * u[j] * v[k];
    
    }
    else if (store_type == TENSOR_STORE_TYPE_SPARSE) {
    
        memset(ret, 0, sizeof(double) * dim);
        for(int p = 0; p < nnz_count; p++) {
        
            int i = idx[0][p], j = idx[1][p], k = idx[2][p];
            
            if (i == j && j == k) {
                ret[i] += values[p] * u[j] * v[k];
            }
            else if (i == j) {
                ret[i] += values[p] * u[i] * v[k] + values[p] * u[k] * v[i];
                ret[k] += values[p] * u[i] * v[i];
            }
            else if (j == k) {
                ret[i] += values[p] * u[j] * v[j];
                ret[j] += values[p] * u[i] * v[j] + values[p] * u[j] * v[i];
            }
            else if (i == k) {
                ret[i] += values[p] * u[j] * v[i] + values[p] * u[i] * v[j];
                ret[j] += values[p] * u[i] * v[i];
            }
            else {
                ret[i] += values[p] * u[j] * v[k] + values[p] * u[k] * v[j];
                ret[j] += values[p] * u[i] * v[k] + values[p] * u[k] * v[i];
                ret[k] += values[p] * u[i] * v[j] + values[p] * u[j] * v[i];
            }
        
        }
    
    }
    else {
        assert(0);
    }

}

double Tensor::sqr_fnorm(bool symeval) {

	double ret = 0;
	
	if (store_type == TENSOR_STORE_TYPE_DENSE) {

        if (!symeval) {
    		for(size_t p = 0; p < dim*dim*dim; p++)
	    		ret += SQR(A[p]);
	    }
	    else {
	        for(size_t i = 0; i < dim; i++)
	            for(size_t j = i; j < dim; j++)
	                for(size_t k = j; k < dim; k++) {
	                    int cnt;
	                    if (i == j && j == k) cnt = 1;
	                    else if (i == j || i == k || j == k) cnt = 3;
	                    else cnt = 6;
	                    ret += cnt * SQR(A[IND3D(i,j,k,dim)]);
	                }
	    }

	}
	else if (store_type == TENSOR_STORE_TYPE_SPARSE) {

		for(int p = 0; p < nnz_count; p++) {

			int i = idx[0][p], j = idx[1][p], k = idx[2][p];
			int t = (i != j) + (j != k);
			int mult = t + (1 << t);
			ret += mult * SQR(A[p]);

		}

	}
	else {
		assert(0);
	}

	return ret;

}

void Tensor::sqr_slice_fnorms(double* fnorms) {

        assert(fnorms);
	
	if (store_type == TENSOR_STORE_TYPE_DENSE) {
		for(size_t i = 0; i < dim; ++i) {
			double res = 0.0;
			size_t idx = i * (size_t) dim * (size_t) dim;
			for(size_t p = 0; p < dim*dim; p++) {
				res += SQR(A[idx + p]);
			}
			fnorms[i] = res;
		}
	}
	else {
		assert(0);
	}
}

void Tensor::slice_stats(double* arr_mean, double* arr_variance) {
	assert(arr_mean);
	assert(arr_variance);
	if (store_type == TENSOR_STORE_TYPE_DENSE) {
		for (size_t i = 0; i < dim; ++i) {
			double sum = 0.0;
			double average, variance;
			size_t idx = i * (size_t) dim * (size_t) dim;
			for (size_t p = 0; p < dim*dim; ++p) {
				sum += A[idx + p];
			}
			average = sum / (dim * dim);
			arr_mean[i] = average;
			sum = 0.0;
			for (size_t p = 0; p < dim*dim; ++p) {
				sum += SQR(A[idx + p] - average);
			}
			variance = sum / (dim * dim);
			arr_variance[i] = variance;
		}
	}
	else {
		assert(0);
	}
}

void Tensor::to_sparse_format() {

	if (store_type == TENSOR_STORE_TYPE_DENSE) {

		nnz_count = 0;
		for(size_t i = 0; i < dim; i++)
			for(size_t j = i; j < dim; j++)
				for(size_t k = j; k < dim; k++)
					if (safe_compare(A[IND3D(i,j,k,dim)], 0))
						nnz_count ++;

//		printf("I\'m at to_sparse_format, nnz_count = %d\n", nnz_count);

		values = new double[nnz_count];
		for(int i = 0; i < 3; i++)
			idx[i] = new int[nnz_count];


		int p = 0;
		for(int i = 0; i < dim; i++)
			for(int j = i; j < dim; j++)
				for(int k = j; k < dim; k++)
					if (safe_compare(A[IND3D(i,j,k,dim)], 0)) {
						idx[0][p] = i;
						idx[1][p] = j;
						idx[2][p] = k;
						values[p++] = A[IND3D(i,j,k,dim)];
					}

//		printf("before del, nnz_count = %d\n", nnz_count);

		delete[] A;
		A = NULL;
		store_type = TENSOR_STORE_TYPE_SPARSE;

//		printf("nnz_count = %d\n", nnz_count);

	}
	else {
		assert(0);
	}

}

void Tensor::sparsify(double rate) {

	if (store_type == TENSOR_STORE_TYPE_DENSE) {

		double mult = 1.0 / rate;

		for(size_t i = 0; i < dim; i++)
			for(size_t j = i; j < dim; j++)
				for(size_t k = j; k < dim; k++)
					if (safe_compare(A[IND3D(i,j,k,dim)], 0)) {
						if ((double)rand() / RAND_MAX < rate)
							A[IND3D(i,j,k,dim)] *= mult;
						else
							A[IND3D(i,j,k,dim)] = 0;
						double t = A[IND3D(i,j,k,dim)];
						A[IND3D(i,k,j,dim)] = t;
						A[IND3D(j,i,k,dim)] = t;
						A[IND3D(j,k,i,dim)] = t;
						A[IND3D(k,i,j,dim)] = t;
						A[IND3D(k,j,i,dim)] = t;
					}

		this->rate = rate;

	}

}

void Tensor::add_rank_one_update(double lambda, double* u) {

	if (store_type == TENSOR_STORE_TYPE_DENSE) {

		for(size_t i = 0; i < dim; i++)
			for(size_t j = 0; j < dim; j++)
				for(size_t k = 0; k < dim; k++)
					A[IND3D(i,j,k,dim)] += lambda * u[i] * u[j] * u[k];

	}
	else if (store_type == TENSOR_STORE_TYPE_SPARSE) {

		double mult = 1.0 / rate;
		for(int p = 0; p < nnz_count; p++) {

			int i = idx[0][p], j = idx[1][p], k = idx[2][p];
			values[p] += mult * lambda * u[i] * u[j] * u[k];

		}

	}
	else {
		assert(0);
	}

}

void Tensor::clear() {

	store_type = TENSOR_STORE_TYPE_NULL;
	if(is_mmapped) {
		if (fd != 0) {
			munmap(memblock, mapping_size);
			close(fd);
		}
	}
	else {
		if (A) {
			delete[] A;
			A = NULL;
		}
	}
	

	if (values) {
		delete[] values;
		values = NULL;
	}

	for(int i = 0; i < 3; i++)
		if (idx[i]) {
			delete[] idx[i];
			idx[i] = NULL;
		}
        
    if (Lambda) delete[] Lambda;
    if (U) delete[] U;

}
