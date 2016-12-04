#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "config.h"
#include "util.h"
#include "corpus.h"
#include "tensor_lda.h"
#include "tensor.h"
#include "hash.h"
#include "count_sketch.h"
#include "fast_tensor_power_method.h"
#include "LDA.h"
#include "matlab_wrapper.h"
#include "fft_wrapper.h"

#ifdef DEBUG_MODE_
Hashes* debug_tensor_lda_hashes[3];
AsymCountSketch* debug_tensor_lda_cs;
CountSketch* debug_tensor_lda_sym_cs;
Hashes* debug_tensor_lda_sym_hashes;
#endif

void compute_word_frequency(Corpus* corpus, double* M1) {

	size_t V = corpus->V;
	int K = corpus->K;

	memset(M1, 0, sizeof(double) * V);

	printf("Computing word frequency ...\n");

	long long tot_num_docs = 0;
	for(int idf = 0; idf < corpus->num_data_files; idf++) {
		
		corpus->load(corpus->data_files[idf]);
		int cnt = 0;
		int thresh = (int)(0.01 * corpus->num_docs);
		for(Document* doc = corpus->docs; doc < corpus->docs + corpus->num_docs; doc ++) {
			if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0; }
			double scale = 1.0 / doc->num_words;
			for(int i = 0; i < doc->num_items; i++) {
				M1[doc->idx[i]] += doc->occs[i] * scale;
				if(doc->idx[i] >= V) {
					printf("doc->idx[%d] = %d out of range (>= %ld) for doc %d\n", i, doc->idx[i], V, cnt);
					assert(doc->idx[i] < V);
				}
			}
		}

		puts("");
		tot_num_docs += corpus->num_docs;

	}

	double scale = 1.0 / tot_num_docs;
	for(int i = 0; i < V; i++)
		M1[i] *= scale;

}

void slow_whiten(Corpus* corpus, double alpha0, double* M1, double* W, double* A2WW, Matlab_wrapper* mat_wrapper) {

	size_t V = corpus->V;
	int K = corpus->K;
	double scale = 0;
	
	printf("Computing word co-occurrence ...\n");

	double* M2 = new double[V*V];
	assert(M2);
	memset(M2, 0, sizeof(double) * V*V);	
	printf("Memory allocation finished.\n");

	// empirical 2-nd order moment

	printf("Computing empirical 2-nd order moment ...\n");

	long long tot_num_docs = 0;
	for(int idf = 0; idf < corpus->num_data_files; idf++) {

		corpus->load(corpus->data_files[idf]);
		int cnt = 0;
		int thresh = (int)(0.01 * corpus->num_docs);
		for(Document* doc = corpus->docs; doc < corpus->docs + corpus->num_docs; doc ++) {
			if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0; }
			if (doc->num_words < 2) {tot_num_docs --; continue;}
			scale = 1.0 / ((double)(doc->num_words) * (doc->num_words - 1));
			for(int i = 0; i < doc->num_items; i++)
				for(int j = 0; j < doc->num_items; j++) {
					int v1 = doc->idx[i], v2 = doc->idx[j];
					int c1 = doc->occs[i], c2 = doc->occs[j];
					if (i == j) {
						M2[IND2D(v1, v2, V)] += scale * c1 * (c2-1);
					}
					else {
						M2[IND2D(v1, v2, V)] += scale * c1 * c2;
					}
				}
		}

		puts("");
		tot_num_docs += corpus->num_docs;

	}
	
	scale = 1.0 / tot_num_docs;
	for(int p = 0; p < SQR(V); p++)
		M2[p] *= scale;

	// sanity check
	double sum = 0;
	for(int p = 0; p < SQR(V); p++)
		sum += M2[p];
	printf("M2 sum = %lf\n", sum);
	assert(safe_compare(sum, 1.0, 1e-6) == 0);

	// add 1-st order correction
	scale = alpha0 / (alpha0 + 1);
	for(int i = 0; i < V; i++)
		for(int j = 0; j < V; j++)
			M2[IND2D(i,j,V)] -= scale * M1[i] * M1[j];

	// whitening: using Matlab svds
	printf("Start whitening M2 ...\n");
	
	printf("Start whitening M2 ...\n");
	double* UL = new double[V*K];
	double* S = new double[K];

	mat_wrapper->eigs(M2, V, K, UL, S);
	printf("sigma_k(M2) = %e\n", S[K-1]);

	double min_Sk = 1e100;
	for(int k = 0; k < K; k++) {
		if (fabs(S[k]) < min_Sk) min_Sk = fabs(S[k]);
		S[k] = 1.0 / sqrt(fabs(S[k]));
	}
	printf("min S[k] = %e\n", min_Sk);
	for(int i = 0; i < V; i++)
		for(int k = 0; k < K; k++)
			W[IND2D(i,k,K)] = UL[IND2D(i,k,K)] * S[k];

	// compute A2WW
	// remove 1-st order correction
	scale = alpha0 / (alpha0 + 1);
	for(int i = 0; i < V; i++)
		for(int j = 0; j < V; j++)
			M2[IND2D(i,j,V)] += scale * M1[i] * M1[j];
	
	double* M2W = new double[V*K];	
	assert(M2W);

	mat_wrapper->multiply(M2, W, V, V, V, K, V, K, false, false, M2W);
	mat_wrapper->multiply(W, M2W, V, K, V, K, K, K, true, false, A2WW);

	delete[] S;
	delete[] UL;

	delete[] M2;
	delete[] M2W;

}

void tensor_lda_preprocess(Corpus* corpus, double* W) {

	// refer to readme.txt for the binary format of tensor LDA

	size_t V = corpus->V;
	int K = corpus->K;
	char filename[1000];
	memset(filename, 0, sizeof(filename));

	double* pw = new double[K];

	for(int idf = 0; idf < corpus->num_data_files; idf ++) {
		
		corpus->load(corpus->data_files[idf]);
		strcpy(filename, corpus->data_files[idf]);
		strcat(filename, ".dat_tensor");
		FILE *fp = fopen(filename, "wb");
		assert(fp);

		fwrite(&(corpus->num_docs), sizeof(int), 1, fp);
		fwrite(&K, sizeof(int), 1, fp);
		for(Document* doc = corpus->docs; doc < corpus->docs + corpus->num_docs; doc ++) {
			memset(pw, 0, sizeof(double) * K);
			for(int i = 0; i < doc->num_items; i++) {
				int v = doc->idx[i], c = doc->occs[i];
				for(int k = 0; k < K; k++)
					pw[k] += c * W[IND2D(v, k, K)];
			}
			fwrite(pw, sizeof(double), K, fp);
			fwrite(&(doc->num_words), sizeof(int), 1, fp);
		}

		fclose(fp);

	}

	delete[] pw;

}

double** tensor_lda_preprocess_load(char* dat_filename, int* num_docs) {

	char filename[1000];
	memset(filename, 0, sizeof(filename));
	
	strcpy(filename, dat_filename);
	strcat(filename, ".dat_tensor");
	FILE* fp = fopen(filename, "rb");
	assert(fp);

	int N = 0, K = 0;
	fread(&N, sizeof(int), 1, fp);
	fread(&K, sizeof(int), 1, fp);
	double** ret = new double*[N];
	assert(ret);

	for(int d = 0; d < N; d++) {
		ret[d] = new double[K+1];
		assert(ret[d]);
		fread(ret[d], sizeof(double), K, fp);
		int t = 0;
		fread(&t, sizeof(int), 1, fp);
		ret[d][K] = t;
	}

	fclose(fp);

	if (num_docs)
		*num_docs = N;
		
	return ret;

}

// W: double V x k
// lambda: double K
// v: double K x K
void tensor_lda_parameter_recovery(LDA* model, double* W, size_t V, int K, double alpha0, double* lambda, double** v, Matlab_wrapper* mat_wrapper) {

	// recover alpha
	double sum = 0;
	for(int k = 0; k < K; k++) {
		model->alpha[k] = 1.0 / (SQR(lambda[k]) + 1e-9);
		sum += model->alpha[k];
	}
	for(int k = 0; k < K; k++) {
		model->alpha[k] *= alpha0 / sum;
	}

	// recover Phi
	double* Winv = new double[K*V];
	assert(Winv);

	mat_wrapper->pinv(W, V, K, Winv);

	for(int k = 0; k < K; k++) {

		double sum = 0;
		for(int i = 0; i < V; i++) {
			model->Phi[k][i] = 0;
			for(int j = 0; j < K; j++)
				model->Phi[k][i] += Winv[IND2D(j,i,V)] * v[k][j];
			if (model->Phi[k][i] < 0) model->Phi[k][i] = 0;
			sum += model->Phi[k][i];
		}
		
		if (safe_compare(sum, 0) == 0) {
			for(int i = 0; i < V; i++)
				model->Phi[k][i] = 1.0 / V;
		}
		else {
			for(int i = 0; i < V; i++)
				model->Phi[k][i] /= sum;
		}

	}

	delete[] Winv;

}

// method: construct M3(W, W, W) explicitly and do naive tensor power method
void slow_tensor_lda(Corpus* corpus, double alpha0, double* W, double* A2WW, LDA* model, Matlab_wrapper* mat_wrapper, int T, int L) {

	size_t V = corpus->V;
	int K = corpus->K;
	double scale = 0, sum = 0;
	long long tot_num_docs = 0;
	
	assert(T > 0);
	assert(L > 0);

	tensor_lda_preprocess(corpus, W);

	double* M1 = new double[V];
	compute_word_frequency(corpus, M1);

	double* M1W = new double[K]; // M1W = W' * M1
	memset(M1W, 0, sizeof(double) * K);
	for(int i = 0; i < V; i++)
		for(int k = 0; k < K; k++)
			M1W[k] += W[IND2D(i,k,K)] * M1[i];

	double* M3WWW = new double[K*K*K];
	assert(M3WWW);
	memset(M3WWW, 0, sizeof(double) * K*K*K);


	// Part 1
	puts("Part 1 ---");

	for(int idf = 0; idf < corpus->num_data_files; idf++) {
	
		int num_docs = 0;
		double** pw = tensor_lda_preprocess_load(corpus->data_files[idf], &num_docs);
		printf("num_docs = %d\n", num_docs);

		int thresh = (int)(0.01 * corpus->num_docs);
		int cnt = 0;

		for(int d = 0; d < num_docs; d++) {
			if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;} 
			int num_words = lround(pw[d][K]);
			if (num_words < 3) {tot_num_docs --; continue;}
			double scale1 = 1.0 / num_words;
			double scale2 = 1.0 / (num_words - 1);
			double scale3 = 1.0 / (num_words - 2);
			for(int k1 = 0; k1 < K; k1++)
				for(int k2 = 0; k2 < K; k2++)
					for(int k3 = 0; k3 < K; k3++)
						M3WWW[IND3D(k1,k2,k3,K)] += (pw[d][k1] * scale1) * (pw[d][k2] * scale2) * (pw[d][k3] * scale3);
		}

		puts("");
		tot_num_docs += num_docs;

		for(int d = 0; d < num_docs; d++)
			delete[] pw[d];
		delete[] pw;

	}


	// Part 2
	puts("Part 2 ---");

	double* P2 = new double[K*K];
	assert(P2);

	for(int idf = 0; idf < corpus->num_data_files; idf ++) {
		
		corpus->load(corpus->data_files[idf]);
		double** pw = tensor_lda_preprocess_load(corpus->data_files[idf], NULL);

		int thresh = (int)(0.01 * corpus->num_docs);
		int cnt = 0;

		int d = 0;
		for(Document* doc = corpus->docs; doc < corpus->docs + corpus->num_docs; doc ++) {
			if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;}
			if (doc->num_words < 3) continue;
			memset(P2, 0, sizeof(double) * K*K);

			for(int i = 0; i < doc->num_items; i++) {
				int v = doc->idx[i], c = doc->occs[i];
				for(int k1 = 0; k1 < K; k1++)
					for(int k2 = 0; k2 < K; k2++)
						P2[IND2D(k1,k2,K)] += c * W[IND2D(v,k1,K)] * W[IND2D(v,k2,K)];
			}

			scale = 1.0 / (doc->num_words - 2);
			for(int k = 0; k < K; k++)
				pw[d][k] *= scale;
			scale = 1.0 / (doc->num_words) / (doc->num_words - 1);
			for(int p = 0; p < K*K; p++)
				P2[p] *= scale;

			for(int k1 = 0; k1 < K; k1++)
				for(int k2 = 0; k2 < K; k2++)
					for(int k3 = 0; k3 < K; k3++)
						M3WWW[IND3D(k1,k2,k3,K)] -= pw[d][k1] * P2[IND2D(k2,k3,K)] + pw[d][k2] * P2[IND2D(k1,k3,K)] + pw[d][k3] * P2[IND2D(k1,k2,K)]; 

			d ++;
		}

		puts("");

		for(int d = 0; d < corpus->num_docs; d++)
			delete[] pw[d];
		delete[] pw;

	}

	delete[] P2;

	scale = 1.0 / tot_num_docs;
	for(int k1 = 0; k1 < K; k1++)
		for(int k2 = 0; k2 < K; k2++)
			for(int k3 = 0; k3 < K; k3++)
				M3WWW[IND3D(k1,k2,k3,K)] *= scale;

	// Part 3
	puts("Part 3 ---");

	double* coeffs = new double[V];
	assert(coeffs);
	memset(coeffs, 0, sizeof(double) * V);

	for(int idf = 0; idf < corpus->num_data_files; idf++) {
		
		corpus->load(corpus->data_files[idf]);
		int num_docs = corpus->num_docs;
		int* pind = new int[num_docs];
		assert(pind);
		memset(pind, 0, sizeof(int) * num_docs);

		for(int i = 0; i < V; i++) {
			int d = 0;
			for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
				while (pind[d] < doc->num_items && doc->idx[pind[d]] < i)
					pind[d] ++;
				if (doc->num_words >= 3 && pind[d] < doc->num_items && doc->idx[pind[d]] == i) {
					double num_words = doc->num_words;
					coeffs[i] += (double)(2 * doc->occs[pind[d]]) / (num_words * (num_words-1) * (num_words-2));
				}
				d ++;
			}
		}

		delete[] pind;

	}

	scale = 1.0 / tot_num_docs;
	for(int k1 = 0; k1 < K; k1++)
		for(int k2 = 0; k2 < K; k2++)
			for(int k3 = 0; k3 < K; k3++)
				for(int i = 0; i < V; i++)
					M3WWW[IND3D(k1,k2,k3,K)] += scale * coeffs[i] * W[IND2D(i,k1,K)] * W[IND2D(i,k2,K)] * W[IND2D(i,k3,K)];

	delete[] coeffs;

	// Part 4
	puts("Part 4 ---");

	for(int k1 = 0; k1 < K; k1++)
		for(int k2 = 0; k2 < K; k2++)
			for(int k3 = 0; k3 < K; k3++)
				M3WWW[IND3D(k1,k2,k3,K)] += 
					-(alpha0/(alpha0+2)) * (A2WW[IND2D(k1,k2,K)] * M1W[k3] + A2WW[IND2D(k1,k3,K)] * M1W[k2] + A2WW[IND2D(k2,k3,K)] * M1W[k1])
					+ (2*SQR(alpha0)/((alpha0+1)*(alpha0+2))) * (M1W[k1] * M1W[k2] * M1W[k3]);

	delete[] M1;
	delete[] M1W;

	printf("Start naive robust tensor power method ...\n");

	Tensor* tensor = new Tensor();
	tensor->store_type = TENSOR_STORE_TYPE_DENSE;
	tensor->dim = K;
	tensor->A = M3WWW;
	tensor->save("slow_tensor_lda.dat");
	double* lambda = new double[K];
	assert(lambda);
	double** v = new double*[K];
	assert(v);
	for(int k = 0; k < K; k++) {
		v[k] = new double[K];
		assert(v[k]);
	}
	
	#ifdef DEBUG_MODE_
	int B = T;
	int b = L;
	Hashes* asym_hashes[3];
	for(int i = 0; i < 3; i++) {
	    asym_hashes[i] = new Hashes(B, b, K, 6);
	    asym_hashes[i]->to_asymmetric_hash();
	}
	AsymCountSketch* asym_cs = new AsymCountSketch(3, asym_hashes);
	asym_cs->set_tensor(tensor);
	for(int i = 0; i < 3; i++) {
	    debug_tensor_lda_hashes[i] = asym_hashes[i];
	}
	debug_tensor_lda_cs = asym_cs;
	
	Hashes* hashes = new Hashes(B, b, K, 6);
	CountSketch* cs_T = new CountSketch(hashes);
	cs_T->set_tensor(tensor, false);
	debug_tensor_lda_sym_hashes = hashes;
	debug_tensor_lda_sym_cs = cs_T;

    #else 
    
	slow_tensor_power_method(tensor, K, K, L, T, lambda, v);
	tensor_lda_parameter_recovery(model, W, V, K, alpha0, lambda, v, mat_wrapper);
	
	for(int k = 0; k < K; k++)
	    model->alpha[k] = alpha0 / K;
	model->alpha0 = alpha0;
	
	#endif
	
	tensor->A = NULL;
	delete[] M3WWW;
	delete tensor;
    
    for(int k = 0; k < K; k++)
        delete[] v[k];
    delete[] v;
    delete[] lambda;

}

void slow_als_lda(Corpus* corpus, double alpha0, double* W, LDA* model, int B, int b, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, int T) {

    size_t V = corpus->V;
    int K = corpus->K;
    double scale = 0;
    printf("slow ALS LDA, V = %ld, K = %d\n", V, K);
    
    double* M1 = new double[V];
	compute_word_frequency(corpus, M1);

    Tensor* tensor = new Tensor(V, TENSOR_STORE_TYPE_DENSE);
    memset(tensor->A, 0, sizeof(double) * V*V*V);
    
    double* A2 = new double[V*V];
    memset(A2, 0, sizeof(double) * V*V);
    
    int tot_num_docs = 0;
//    double multiplier = 10 * exp(1.5 * log(V));
    double multiplier = 1;
    
    for(int idf = 0; idf < corpus->num_data_files; idf++) {
    
        corpus->load(corpus->data_files[idf]);
        int num_docs = corpus->num_docs;
        int thresh = (int)(0.01 * num_docs);
        int cnt = 0;
        
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        
            if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;}
            int m = doc->num_words;
            scale = multiplier / ((double)m * (m-1) * (m-2));
            
            for(int i = 0; i < doc->num_items; i++)
                for(int j = 0; j < doc->num_items; j++)
                    for(int k = 0; k < doc->num_items; k++) {
                        int v1 = doc->idx[i], v2 = doc->idx[j], v3 = doc->idx[k];
                        int c1 = doc->occs[i], c2 = doc->occs[j], c3 = doc->occs[k];
                        
                        // Part 1
                        if (v1 == v2 && v2 == v3) {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * (c1-1) * (c1-2);
                        }
                        else if (v1 == v2) {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * (c1-1) * c3;
                        }
                        else if (v1 == v3) {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * (c1-1) * c2;
                        }
                        else if (v2 == v3) {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * c2 * (c2-1);
                        }
                        else {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * c2 * c3;
                        }                        
                    }
            
            scale = 1.0 / ((double)m * (m-1));
            for(int i = 0; i < doc->num_items; i++)
                for(int j = 0; j < doc->num_items; j++) {
                    int v1 = doc->idx[i], v2 = doc->idx[j];
                    int c1 = doc->occs[i], c2 = doc->occs[j];
                    if (v1 == v2) {
                        A2[IND2D(v1,v2,V)] += scale * c1 * (c1-1);
                    }
                    else {
                        A2[IND2D(v1,v2,V)] += scale * c1 * c2;
                    }
                }
            

            
        }
        
        puts("");
        tot_num_docs += num_docs;        
    
    }
    
        // normalize
        scale = 1.0 / tot_num_docs;
        for(size_t p = 0; p < V*V*V; p++)
            tensor->A[p] *= scale;
        for(int p = 0; p < V*V; p++)
            A2[p] *= scale;
            
        // Part 2
        scale = multiplier * alpha0 / (alpha0 + 2);
        for(int i = 0; i < V; i++)
            for(int j = 0; j < V; j++)
                for(int k = 0; k < V; k++)
                    tensor->A[IND3D(i,j,k,V)] -= scale * (A2[IND2D(i,j,V)]*M1[k] + A2[IND2D(i,k,V)]*M1[j] + A2[IND2D(j,k,V)]*M1[i]);
        
        // Part 3
        scale = multiplier * 2 * SQR(alpha0) / ((alpha0+1) * (alpha0+2));
        for(int i = 0; i < V; i++)
            for(int j = 0; j < V; j++)
                for(int k = 0; k < V; k++)
                    tensor->A[IND3D(i,j,k,V)] += scale * M1[i] * M1[j] * M1[k];
                    
        // ALS
        double* lambda = new double[K];
        double* U = new double[K*V];
        double* U2 = new double[K*V];
        double* U3 = new double[K*V];
        
       /* for(int k = 0; k < K; k++) {
            generate_uniform_sphere_point(V, U + k * V);
            for(int i = 0; i < V; i++)
                U[IND2D(k,i,V)] = fabs(U[IND2D(k,i,V)]);
            memcpy(U2 + k * V, U + k * V, sizeof(double) * V);
            memcpy(U3 + k * V, U + k * V, sizeof(double) * V);
        }*/
        
        for(int k = 0; k < K; k++) {
            double sum = 0;
            for(int i = 0; i < V; i++) {
                U[IND2D(k,i,V)] = W[IND2D(i,k,K)];
                sum += SQR(U[IND2D(k,i,V)]);
            }
            scale = 1.0 / sqrt(sum);
            for(int i = 0; i < V; i++)
                U[IND2D(k,i,V)] *= scale;
            memcpy(U2 + k * V, U + k * V, sizeof(double) * V);
            memcpy(U3 + k * V, U + k * V, sizeof(double) * V);
        }
        
        /*for(int k = 0; k < K; k++) {
            memcpy(U + k * V, model->Phi[k], sizeof(double) * V);
            double sum = 0;
            for(int i = 0; i < V; i++)
                sum += SQR(U[IND2D(k,i,V)]);
            double scale = 1.0 / sqrt(sum);
            for(int i = 0; i < V; i++)
                U[IND2D(k,i,V)] *= scale;
        }*/
        
        /*for(int i1 = 0; i1 < V; i1++)
            for(int i2 = 0; i2 < V; i2++)
                for(int i3 = 0; i3 < V; i3++)
                    for(int k = 0; k < K; k++)
                        tensor->A[IND3D(i1,i2,i3,V)] += model->Phi[k][i1] * model->Phi[k][i2] * model->Phi[k][i3];*/
        
        /*scale = 1.0 / sqrt(tensor->sqr_fnorm());
        for(int p = 0; p < V*V*V; p++)
            tensor->A[p] *= scale;*/
        
        printf("tensor.fnorm = %lf\n", tensor->sqr_fnorm());
        //slow_ALS(tensor, V, K, T, mat_wrapper, lambda, U);
        //slow_asym_ALS(tensor, V, K, T, mat_wrapper, lambda, U, U2, U3);
        
        Hashes* asym_hashes[3];
        for(int i = 0; i < 3; i++) {
            asym_hashes[i] = new Hashes(B, b, V, 6);
            asym_hashes[i]->to_asymmetric_hash();
        }
        AsymCountSketch* asym_cs = new AsymCountSketch(3, asym_hashes);
        asym_cs->set_tensor(tensor);
        
        double residue = fast_asym_ALS(asym_cs, V, K, T, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, U, U2, U3, true);
		printf("normalized residue = %lf\n", residue);
        
        for(int i1 = 0; i1 < V; i1++)
            for(int i2 = 0; i2 < V; i2++)
                for(int i3 = 0; i3 < V; i3++)
                    for(int k = 0; k < K; k++)
                        tensor->A[IND3D(i1,i2,i3,V)] -= lambda[k] * U[IND2D(k,i1,V)] * U2[IND2D(k,i2,V)] * U3[IND2D(k,i3,V)];
        printf("tensor.fnorm = %lf\n", tensor->sqr_fnorm());
        
        for(int k = 0; k < K; k++) {
            double sum = 0;
            for(int i = 0; i < V; i++) {
                if (U[IND2D(k,i,V)] < 0) U[IND2D(k,i,V)] = 0;
                sum += U[IND2D(k,i,V)];
            }
            if (safe_compare(sum, 0) == 0) {
                for(int i = 0; i < V; i++)
                    U[IND2D(k,i,V)] = 1.0 / V;
            }
            else {
                scale = 1.0 / sum;
                for(int i = 0; i < V; i++)
                    U[IND2D(k,i,V)] *= scale;
            }
        }
        
        for(int k = 0; k < K; k++)
            memcpy(model->Phi[k], U + k * V, sizeof(double) * V);   

    for(int k = 0; k < K; k++)
        model->alpha[k] = alpha0 / K;
    
    delete tensor;
    delete[] M1;
    delete[] A2;
    delete[] U;
    delete[] U2;
    delete[] U3;
    
    #ifdef DEBUG_MODE_
    for(int i = 0; i < 3; i++)
        debug_tensor_lda_hashes[i] = asym_hashes[i];
    debug_tensor_lda_cs = asym_cs;
    #endif
    

}

AsymCountSketch* fast_tensor_lda(Corpus* corpus, double alpha0, int B, int b, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper) {
    
    size_t V = corpus->V;
    int K = corpus->K;
    printf("fast tensor LDA: using count sketch B = %d, b = %d, V = %ld\n", B, b, V);
    
    double* values = new double[V];
    double scale = 0;
    long long num_tot_docs = 0;
    
    int* ind3d = new int[3];
    
    double* M1 = new double[V];
	compute_word_frequency(corpus, M1);
    int* indM1 = new int[V];
    for(int i = 0; i < V; i++)
        indM1[i] = i;
    
    // Initialize: count sketch
    Hashes* asym_hashes[3];
    #ifdef DEBUG_MODE_
        for(int i = 0; i < 3; i++)
            asym_hashes[i] = debug_tensor_lda_hashes[i];
    #else
        for(int i = 0; i < 3; i++) {
            asym_hashes[i] = new Hashes(B, b, V, 6);
            asym_hashes[i]->to_asymmetric_hash();
        }
    #endif
    
    AsymCountSketch* asym_cs = new AsymCountSketch(3, asym_hashes);
    
    AsymCountSketch* cs_u1 = new AsymCountSketch(asym_hashes[0]);
    AsymCountSketch* cs_u2 = new AsymCountSketch(asym_hashes[1]);
    AsymCountSketch* cs_u3 = new AsymCountSketch(asym_hashes[2]);
    
    AsymCountSketch* m1_cs_u1 = new AsymCountSketch(asym_hashes[0]);
    AsymCountSketch* m1_cs_u2 = new AsymCountSketch(asym_hashes[1]);
    AsymCountSketch* m1_cs_u3 = new AsymCountSketch(asym_hashes[2]);
    m1_cs_u1->set_vector(M1, V);
    m1_cs_u2->set_vector(M1, V);
    m1_cs_u3->set_vector(M1, V);
    m1_cs_u1->fft(fft_wrapper);
    m1_cs_u2->fft(fft_wrapper);
    m1_cs_u3->fft(fft_wrapper);
    
    Hashes* pair_hashes[2];
    pair_hashes[0] = asym_hashes[0]; pair_hashes[1] = asym_hashes[1];
    AsymCountSketch* cs_u12 = new AsymCountSketch(2, pair_hashes);
    pair_hashes[0] = asym_hashes[0]; pair_hashes[1] = asym_hashes[2];
    AsymCountSketch* cs_u13 = new AsymCountSketch(2, pair_hashes);
    pair_hashes[0] = asym_hashes[1]; pair_hashes[1] = asym_hashes[2];
    AsymCountSketch* cs_u23 = new AsymCountSketch(2, pair_hashes);
    
    //double multiplier = 10 * exp(1.5 * log(V));
    double multiplier = 1;
    
    for(int idf = 0; idf < corpus->num_data_files; idf ++) {
    
        corpus->load(corpus->data_files[idf]);
        int num_docs = corpus->num_docs;
        int thresh = (int)(0.01 * num_docs);
        int cnt = 0;
        
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        
            if (cnt ++ > thresh) {putchar('.'); fflush(stdout); cnt = 0; }
            int m = doc->num_words;
            if (m < 3) {num_tot_docs --; continue;}
            scale = multiplier / ((double)m * (m-1) * (m-2));
            
            // n \tensor n \tensor n
            for(int i = 0; i < doc->num_items; i++)
                values[i] = doc->occs[i];
            cs_u1->set_sparse_vector(doc->idx, values, doc->num_items);
            cs_u2->set_sparse_vector(doc->idx, values, doc->num_items);
            cs_u3->set_sparse_vector(doc->idx, values, doc->num_items);
            cs_u1->fft(fft_wrapper);
            cs_u2->fft(fft_wrapper);
            cs_u3->fft(fft_wrapper);
            asym_cs->add_rank_one_tensor(scale, V, cs_u1, cs_u2, cs_u3, fft_wrapper, ifft_wrapper);
            
            // diag(n) \tensor n
            cs_u12->set_sparse_vector(doc->idx, values, doc->num_items);
            cs_u13->set_sparse_vector(doc->idx, values, doc->num_items);
            cs_u23->set_sparse_vector(doc->idx, values, doc->num_items);
            cs_u12->fft(fft_wrapper);
            cs_u13->fft(fft_wrapper);
            cs_u23->fft(fft_wrapper);
            asym_cs->add_rank_one_tensor(-scale, V, cs_u12, cs_u3, fft_wrapper, ifft_wrapper);
            asym_cs->add_rank_one_tensor(-scale, V, cs_u13, cs_u2, fft_wrapper, ifft_wrapper);
            asym_cs->add_rank_one_tensor(-scale, V, cs_u23, cs_u1, fft_wrapper, ifft_wrapper);
            
            // T(i,i,i) = n_i
            for(int i = 0; i < doc->num_items; i++) {
                ind3d[0] = ind3d[1] = ind3d[2] = doc->idx[i];
                double value = 2 * doc->occs[i];
                asym_cs->add_entry(ind3d, scale * value);
            }
                
            // n \tensor n \tensor M1
            scale = multiplier * alpha0 / (alpha0+2) / (double(m) * (m-1));
            asym_cs->add_rank_one_tensor(-scale, V, cs_u1, cs_u2, m1_cs_u3, fft_wrapper, ifft_wrapper);
            asym_cs->add_rank_one_tensor(-scale, V, cs_u1, m1_cs_u2, cs_u3, fft_wrapper, ifft_wrapper);
            asym_cs->add_rank_one_tensor(-scale, V, m1_cs_u1, cs_u2, cs_u3, fft_wrapper, ifft_wrapper);
            
            // diag(n) \tensor M1
            asym_cs->add_rank_one_tensor(scale, V, cs_u12, m1_cs_u3, fft_wrapper, ifft_wrapper);
            asym_cs->add_rank_one_tensor(scale, V, cs_u13, m1_cs_u2, fft_wrapper, ifft_wrapper);
            asym_cs->add_rank_one_tensor(scale, V, cs_u23, m1_cs_u1, fft_wrapper, ifft_wrapper);
            
        }
        
        num_tot_docs += num_docs;
    
    }
    
    puts("");
    
    // normalize
    scale = 1.0 / num_tot_docs;
    for(int d = 0; d < B; d++)
        for(int i = 0; i < POWER2(b); i++) {
            asym_cs->cs[d][i][0] *= scale;
            asym_cs->cs[d][i][1] *= scale;
        }
        
    // M1 \tensor M1 \tensor M1
    scale = multiplier * 2*SQR(alpha0) / ((alpha0+1) * (alpha0+2));
    asym_cs->add_rank_one_tensor(scale, V, m1_cs_u1, m1_cs_u2, m1_cs_u3, fft_wrapper, ifft_wrapper);
    
    #ifdef DEBUG_MODE_
        for(int d = 0; d < B; d++)
            for(int i = 0; i < POWER2(b); i++) {
                if (safe_compare(asym_cs->cs[d][i][0], debug_tensor_lda_cs->cs[d][i][0]) != 0 ||
                    safe_compare(asym_cs->cs[d][i][1], debug_tensor_lda_cs->cs[d][i][1]) != 0) {
                
                    printf("d = %d, i = %d\n", d, i);
                    printf("ref: (%lf, %lf)\n", debug_tensor_lda_cs->cs[d][i][0], debug_tensor_lda_cs->cs[d][i][1]);
                    printf("(%lf, %lf)\n", asym_cs->cs  [d][i][0], asym_cs->cs[d][i][1]);
                    assert(0);
                
                }
            }
    #endif
    
    delete[] values;
    delete[] ind3d;
    delete[] M1;
    delete[] indM1;
    
    return asym_cs;
    
}

double slow_tensor_kernel_eval_tuuu(Corpus* corpus, double alpha0, double* u) {

    size_t V = corpus->V;
    int K = corpus->K;
    double scale = 0;
    
    double* M1 = new double[V];
	compute_word_frequency(corpus, M1);

    Tensor* tensor = new Tensor(V, TENSOR_STORE_TYPE_DENSE);
    memset(tensor->A, 0, sizeof(double) * V*V*V);
    
    double* A2 = new double[V*V];
    memset(A2, 0, sizeof(double) * V*V);
    
    double multiplier = 10 * exp(1.5 * log(V));    
    long long tot_num_docs = 0;
    for(int idf = 0; idf < corpus->num_data_files; idf++) {
    
        corpus->load(corpus->data_files[idf]);
        int num_docs = corpus->num_docs;
        int thresh = (int)(0.01 * num_docs);
        int cnt = 0;
        
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        
            if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;}
            int m = doc->num_words;
            if (m < 3) {tot_num_docs --; continue; }
            scale = multiplier / ((double)m * (m-1) * (m-2));
            
            for(int i = 0; i < doc->num_items; i++)
                for(int j = 0; j < doc->num_items; j++)
                    for(int k = 0; k < doc->num_items; k++) {
                        int v1 = doc->idx[i], v2 = doc->idx[j], v3 = doc->idx[k];
                        int c1 = doc->occs[i], c2 = doc->occs[j], c3 = doc->occs[k];
                        
                        // Part 1
                        if (v1 == v2 && v2 == v3) {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * (c1-1) * (c1-2);
                        }
                        else if (v1 == v2) {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * (c1-1) * c3;
                        }
                        else if (v1 == v3) {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * (c1-1) * c2;
                        }
                        else if (v2 == v3) {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * c2 * (c2-1);
                        }
                        else {
                            tensor->A[IND3D(v1,v2,v3,V)] += scale * c1 * c2 * c3;
                        }                        
                    }
            
            scale = 1.0 / ((double)m * (m-1));
            for(int i = 0; i < doc->num_items; i++)
                for(int j = 0; j < doc->num_items; j++) {
                    int v1 = doc->idx[i], v2 = doc->idx[j];
                    int c1 = doc->occs[i], c2 = doc->occs[j];
                    if (v1 == v2) {
                        A2[IND2D(v1,v2,V)] += scale * c1 * (c1-1);
                    }
                    else {
                        A2[IND2D(v1,v2,V)] += scale * c1 * c2;
                    }
                }
            

            
        }
        
        puts("");
        tot_num_docs += num_docs;        
    
    }
    
    // normalize
    scale = 1.0 / tot_num_docs;
    for(size_t p = 0; p < V*V*V; p++)
        tensor->A[p] *= scale;
    for(int p = 0; p < V*V; p++)
        A2[p] *= scale;
            
    // Part 2
    scale = multiplier * alpha0 / (alpha0 + 2);
    for(int i = 0; i < V; i++)
        for(int j = 0; j < V; j++)
            for(int k = 0; k < V; k++)
                tensor->A[IND3D(i,j,k,V)] -= scale * (A2[IND2D(i,j,V)]*M1[k] + A2[IND2D(i,k,V)]*M1[j] + A2[IND2D(j,k,V)]*M1[i]);
        
    // Part 3
    scale = multiplier * 2 * SQR(alpha0) / ((alpha0+1) * (alpha0+2));
    for(int i = 0; i < V; i++)
        for(int j = 0; j < V; j++)
            for(int k = 0; k < V; k++)
               tensor->A[IND3D(i,j,k,V)] += scale * M1[i] * M1[j] * M1[k];
    
    return tensor->Tuuu(u);

}

double fast_tensor_kernel_eval_tuuu(Corpus* corpus, double alpha0, double* u) {

    size_t V = corpus->V;
    int K = corpus->K;
    double ret = 0;
    double scale = 0, scale1 = 0, scale2 = 0;
    
    double* M1 = new double[V];
    compute_word_frequency(corpus, M1);
    double sum_m1 = 0;
    for(int i = 0; i < V; i++)
       sum_m1 += u[i] * M1[i];    
    
   // double multiplier = 10 * exp(1.5 * log(V));
    double multiplier = 1;
    
    long long tot_num_docs = 0;
    for(int idf = 0; idf < corpus->num_data_files; idf++) {
    
        corpus->load(corpus->data_files[idf]);
        int num_docs = corpus->num_docs;
        int cnt = 0;
        int thresh = (int)(0.01 * num_docs);
        
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        
            if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;}
            int m = doc->num_words;
            if (m < 3) { tot_num_docs --; continue; }
            scale = multiplier / ((double)m * (m-1) * (m-2));
            
            // n \tensor n \tensor n
            double sum = 0;
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                sum += c * u[p];
            }
            ret += scale * sum * sum * sum;
            
            // diag(n) \tensor n
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                ret -= 3 * scale * c * u[p] * u[p] * sum;
            }
            
            // T(i,i,i) = n_i
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                ret += 2 * scale * c * u[p] * u[p] * u[p];
            }
                
            // n \tensor n \tensor M1
            scale = multiplier * alpha0 / (alpha0+2) / ((double)m * (m-1));
            ret -= 3 * scale * sum * sum * sum_m1;
        
            // diag(n) \tensor M1
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                ret += 3 * scale * c * u[p] * u[p] * sum_m1;
            }
        
        }
        
        puts("");
        tot_num_docs += num_docs;
    
    }
    
    // normalize
    scale = 1.0 / tot_num_docs;
    ret *= scale;
        
    // M1 \tensor M1 \tensor M1
    scale = multiplier * 2*SQR(alpha0) / ((alpha0+1) * (alpha0+2));
    ret += scale * sum_m1 * sum_m1 * sum_m1;
    
    delete[] M1;
    return ret;

}

void fast_whiten_tensor_lda(int method, Corpus* corpus, double alpha0, double* W, LDA* model, int B, int b, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, int T, int L) {

    size_t V = corpus->V;
    int K = corpus->K;
    double scale = 0, scale1 = 0, scale2 = 0;
    assert(method == 0 || method == 1);
    assert(T > 0 && L > 0);
    
    printf("fast whiten tensor LDA: method = %d, B = %d, b = %d, T = %d, L = %d\n", method, B, b, T, L);
    
    clock_t t1, t2;
    
    t1 = clock();

    long long tot_num_docs = 0;
    //double multiplier = 10 * exp(1.5 * log(V));    
    double multiplier = 1.0;
    
    double* M1 = new double[V];
	compute_word_frequency(corpus, M1);    
	double* M1W = new double[K]; // M1W = W' * M1
	memset(M1W, 0, sizeof(double) * K);
	for(int i = 0; i < V; i++)
		for(int k = 0; k < K; k++)
			M1W[k] += W[IND2D(i,k,K)] * M1[i];	
    
    Hashes* asym_hashes[3];
    #ifdef DEBUG_MODE_
    for(int i = 0; i < 3; i++)
        asym_hashes[i] = debug_tensor_lda_hashes[i];
    #else
    for(int i = 0; i < 3; i++) {
        asym_hashes[i] = new Hashes(B, b, K, 6);
        asym_hashes[i]->to_asymmetric_hash();
    }
    #endif
    
    AsymCountSketch* asym_cs = new AsymCountSketch(3, asym_hashes);
    AsymCountSketch* asym_u[3];
    AsymCountSketch* asym_m1[3];
    AsymCountSketch* asym_w[3];
    for(int i = 0; i < 3; i++) {
        asym_m1[i] = new AsymCountSketch(asym_hashes[i]);
        asym_u[i] = new AsymCountSketch(asym_hashes[i]);
        asym_w[i] = new AsymCountSketch(asym_hashes[i]);
    }
    
    for(int i = 0; i < 3; i++) {
        asym_m1[i]->set_vector(M1W, K);
        asym_m1[i]->fft(fft_wrapper);
    }
    
    double* coeff1 = new double[V]; // w_i \tensor w_i \tensor w_i
    double* coeff2 = new double[V]; // w_i \tensor w_i \tensor m_1
    double* sumwn = new double[V * K];
    memset(coeff1, 0, sizeof(double) * V);
    memset(coeff2, 0, sizeof(double) * V);
    memset(sumwn, 0, sizeof(double) * V*K);
    
    double* pw = new double[K];
    
    // Part 1
    puts("Part 1 ---");
    
    for(int idf = 0; idf < corpus->num_data_files; idf ++) {
    
        corpus->load(corpus->data_files[idf]);
        int num_docs = corpus->num_docs;
        int thresh = (int)(0.01 * num_docs);
        int cnt = 0;
        
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        
            if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;}
            int m = doc->num_words;
            if (m < 3) {tot_num_docs --; continue;}
            scale1 = multiplier / ((double)m * (m-1) * (m-2));
            scale2 = multiplier * alpha0 / ((alpha0+2) * m * (m-1));
            
            memset(pw, 0, sizeof(double) * K);
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                for(int k = 0; k < K; k++)
                    pw[k] += c * W[IND2D(p,k,K)];
            }
            
            for(int i = 0; i < 3; i++) {
                asym_u[i]->set_vector(pw, K);
                asym_u[i]->fft(fft_wrapper);
            }
            asym_cs->add_rank_one_tensor(scale1, K, asym_u[0], asym_u[1], asym_u[2], fft_wrapper, ifft_wrapper, false);
            
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                coeff1[p] += scale1 * 2 * c;
                coeff2[p] += scale2 * c;
                for(int k = 0; k < K; k++)
                    sumwn[IND2D(p,k,K)] += scale1 * c * pw[k];
            }
            
            asym_cs->add_rank_one_tensor(-scale2, K, asym_u[0], asym_u[1], asym_m1[2], fft_wrapper, ifft_wrapper, false);
            asym_cs->add_rank_one_tensor(-scale2, K, asym_u[0], asym_m1[1], asym_u[2], fft_wrapper, ifft_wrapper, false);
            asym_cs->add_rank_one_tensor(-scale2, K, asym_m1[0], asym_u[1], asym_u[2], fft_wrapper, ifft_wrapper, false);
        
        }
        
        puts("");
        tot_num_docs += num_docs;
    
    }
    
    // Part 2
    puts("Part 2 ---");
    
    int thresh = (int)(0.01 * V);
    int cnt = 0;
    for(int i = 0; i < V; i++) {
        if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;}
        for(int j = 0; j < 3; j++) {
            asym_w[j]->set_vector(W + i * K, K);
            asym_w[j]->fft(fft_wrapper);
            asym_u[j]->set_vector(sumwn + i * K, K);
            asym_u[j]->fft(fft_wrapper);
        }
        // wi \tensor wi \tensor wn
        asym_cs->add_rank_one_tensor(-1, K, asym_w[0], asym_w[1], asym_u[2], fft_wrapper, ifft_wrapper, false);
        asym_cs->add_rank_one_tensor(-1, K, asym_w[0], asym_u[1], asym_w[2], fft_wrapper, ifft_wrapper, false);
        asym_cs->add_rank_one_tensor(-1, K, asym_u[0], asym_w[1], asym_w[2], fft_wrapper, ifft_wrapper, false);
        // wi \tensor wi \tensor wi
        asym_cs->add_rank_one_tensor(coeff1[i], K, asym_w[0], asym_w[1], asym_w[2], fft_wrapper, ifft_wrapper, false);
        // wi \tensor wi \tensor M1
        asym_cs->add_rank_one_tensor(coeff2[i], K, asym_w[0], asym_w[1], asym_m1[2], fft_wrapper, ifft_wrapper, false);
        asym_cs->add_rank_one_tensor(coeff2[i], K, asym_w[0], asym_m1[1], asym_w[2], fft_wrapper, ifft_wrapper, false);
        asym_cs->add_rank_one_tensor(coeff2[i], K, asym_m1[0], asym_w[1], asym_w[2], fft_wrapper, ifft_wrapper, false);
    }
    puts("");
    
    asym_cs->fft(ifft_wrapper);
    
    // normalize
    scale = 1.0 / tot_num_docs;
    for(int d = 0; d < B; d++)
        for(int i = 0; i < POWER2(b); i++) {
            asym_cs->cs[d][i][0] *= scale;
            asym_cs->cs[d][i][1] *= scale;
        }
        
    // Part 3
    puts("Part 3 ---");
    scale = multiplier * 2 * SQR(alpha0) / ((alpha0+1) * (alpha0+2));
    asym_cs->add_rank_one_tensor(scale, K, asym_m1[0], asym_m1[1], asym_m1[2], fft_wrapper, ifft_wrapper);
    
    #ifdef DEBUG_MODE_
    for(int d = 0; d < B; d++)
        for(int i = 0; i < POWER2(b); i++) {
            //printf("(%lf %lf), (%lf %lf)\n", asym_cs->cs[d][i][0], asym_cs->cs[d][i][1], debug_tensor_lda_cs->cs[d][i][0], debug_tensor_lda_cs->cs[d][i][1]);
            assert(safe_compare(asym_cs->cs[d][i][0], debug_tensor_lda_cs->cs[d][i][0]) == 0);
            assert(safe_compare(asym_cs->cs[d][i][1], debug_tensor_lda_cs->cs[d][i][1]) == 0);
        }
    #endif
    
    t2 = clock();
    printf("Building count sketch time: %lf\n", 1e-6 * (t2 - t1));
    
    if (method == 0) {
    
        // ALS
        double* lambda = new double[K];
        double* AA = new double[K*K];
        double* BB = new double[K*K];
        double* CC = new double[K*K];
        
        for(int k = 0; k < K; k++) {
            generate_uniform_sphere_point(K, AA + k * K);
            memcpy(BB + k * K, AA + k * K, sizeof(double) * K);
            memcpy(CC + k * K, AA + k * K, sizeof(double) * K);
        }
        
        double residue = fast_asym_ALS(asym_cs, K, K, T, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, AA, BB, CC, true);
        printf("residue = %lf\n", residue);
        
        double** v = new double*[K];
        for(int k = 0; k < K; k++) {
            v[k] = new double[K];
            memcpy(v[k], AA + k * K, sizeof(double) * K);
        }
     	tensor_lda_parameter_recovery(model, W, V, K, alpha0, lambda, v, mat_wrapper);
     	
     	delete[] lambda;
     	delete[] AA;
     	delete[] BB;
     	delete[] CC;
     	for(int k = 0; k < K; k++)
     	    delete[] v[k];
     	delete[] v;
    
    }
    else {
    
        // robust tensor power method
        double* lambda = new double[K];
        double** v = new double*[K];
        for(int k = 0; k < K; k++)
            v[k] = new double[K];
            
        double residue = fast_asym_tensor_power_method(asym_cs, K, K, L, T, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, v, true);
        printf("residue = %lf\n", residue);
        
        tensor_lda_parameter_recovery(model, W, V, K, alpha0, lambda, v, mat_wrapper);
            
        delete[] lambda;
        for(int k = 0; k < K; k++)
            delete[] v[k];
        delete[] v;
    
    }
    
    for(int k = 0; k < K; k++)
        model->alpha[k] = alpha0 / K;
    model->alpha0 = alpha0;
    
    for(int i = 0; i < 3; i++) {
        delete asym_hashes[i];
        delete asym_u[i];
        delete asym_m1[i];
        delete asym_w[i];
    }
    delete asym_cs;
    delete[] coeff1;
    delete[] coeff2;
    delete[] sumwn;
    delete[] pw;

}

void fast_symmetric_whiten_tensor_lda(int method, Corpus* corpus, double alpha0, double* W, LDA* model, int B, int b, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, int T, int L) {

    size_t V = corpus->V;
    int K = corpus->K;
    double scale = 0, scale1 = 0, scale2 = 0;
    assert(method == 0 || method == 1);
    assert(T > 0 && L > 0);
    
    printf("fast whiten tensor LDA: method = %d, B = %d, b = %d, T = %d, L = %d\n", method, B, b, T, L);
    
    clock_t t1, t2;
    
    t1 = clock();
    
    long long tot_num_docs = 0;
    double multiplier = 1.0;
    
    double* M1 = new double[V];
	compute_word_frequency(corpus, M1);    
	double* M1W = new double[K]; // M1W = W' * M1
	memset(M1W, 0, sizeof(double) * K);
	for(int i = 0; i < V; i++)
		for(int k = 0; k < K; k++)
			M1W[k] += W[IND2D(i,k,K)] * M1[i];	
    
    Hashes* hashes;
    #ifdef DEBUG_MODE_
    hashes = debug_tensor_lda_sym_hashes;
    #else
    hashes = new Hashes(B, b, K, 6);
    #endif
    
    CountSketch* cs_T = new CountSketch(hashes);
    CountSketch* cs_u = new CountSketch(hashes);
    CountSketch* cs_w = new CountSketch(hashes);
    CountSketch* cs_m1 = new CountSketch(hashes);     
    cs_m1->set_vector(M1W, K, 1);
    cs_m1->fft(fft_wrapper);
    
    double* coeff1 = new double[V]; // w_i \tensor w_i \tensor w_i
    double* coeff2 = new double[V]; // w_i \tensor w_i \tensor m_1
    double* sumwn = new double[V * K];
    memset(coeff1, 0, sizeof(double) * V);
    memset(coeff2, 0, sizeof(double) * V);
    memset(sumwn, 0, sizeof(double) * V*K);
    
    double* pw = new double[K];
    
    // Part 1
    puts("Part 1 ---");
    
    for(int idf = 0; idf < corpus->num_data_files; idf ++) {
    
        corpus->load(corpus->data_files[idf]);
        int num_docs = corpus->num_docs;
        int thresh = (int)(0.01 * num_docs);
        int cnt = 0;
        
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        
            if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;}
            int m = doc->num_words;
            if (m < 3) {tot_num_docs --; continue;}
            scale1 = multiplier / ((double)m * (m-1) * (m-2));
            scale2 = multiplier * alpha0 / ((alpha0+2) * m * (m-1));
            
            memset(pw, 0, sizeof(double) * K);
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                for(int k = 0; k < K; k++)
                    pw[k] += c * W[IND2D(p,k,K)];
            }
            
            cs_u->set_vector(pw, K, 1);
            cs_u->fft(fft_wrapper);
            cs_T->add_rank_one_tensor(scale1, cs_u);
            
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                coeff1[p] += scale1 * 2 * c;
                coeff2[p] += scale2 * c;
                for(int k = 0; k < K; k++)
                    sumwn[IND2D(p,k,K)] += scale1 * c * pw[k];
            }
            
            cs_T->add_asym_rank_one_tensor(-scale2, cs_u, cs_m1);
        
        }
        
        puts("");
        tot_num_docs += num_docs;
    
    }
    
    // Part 2
    puts("Part 2 ---");
    
    int thresh = (int)(0.01 * V);
    int cnt = 0;
    for(int i = 0; i < V; i++) {
        if (cnt++ > thresh) {putchar('.'); fflush(stdout); cnt = 0;}
        cs_w->set_vector(W + i * K, K, 1);
        cs_w->fft(fft_wrapper);
        cs_u->set_vector(sumwn + i * K, K, 1);
        cs_u->fft(fft_wrapper);
        // wi \tensor wi \tensor wn
        cs_T->add_asym_rank_one_tensor(-1, cs_w, cs_u);
        // wi \tensor wi \tensor wi
        cs_T->add_rank_one_tensor(coeff1[i], cs_w);
        // wi \tensor wi \tensor M1
        cs_T->add_asym_rank_one_tensor(coeff2[i], cs_w, cs_m1);
    }
    puts("");
    
    cs_T->fft(ifft_wrapper);
    
    // normalize
    scale = 1.0 / tot_num_docs;
    for(int d = 0; d < B; d++)
        for(int i = 0; i < POWER2(b); i++) {
            cs_T->cs[d][i][0] *= scale;
            cs_T->cs[d][i][1] *= scale;
        }
        
    // Part 3
    puts("Part 3 ---");
    scale = multiplier * 2 * SQR(alpha0) / ((alpha0+1) * (alpha0+2));
    cs_T->add_rank_one_tensor(scale, M1W, K, fft_wrapper, ifft_wrapper, false);
    
    #ifdef DEBUG_MODE_
    for(int d = 0; d < B; d++)
        for(int i = 0; i < POWER2(b); i++) {
          //  printf("(%lf %lf), (%lf %lf)\n", cs_T->cs[d][i][0], cs_T->cs[d][i][1], debug_tensor_lda_sym_cs->cs[d][i][0], debug_tensor_lda_sym_cs->cs[d][i][1]); 
            assert(safe_compare(cs_T->cs[d][i][0], debug_tensor_lda_sym_cs->cs[d][i][0]) == 0);
            assert(safe_compare(cs_T->cs[d][i][1], debug_tensor_lda_sym_cs->cs[d][i][1]) == 0);
        }
    return;
    #endif
    
    t2 = clock();
    printf("Building count sketch time: %lf\n", (1e-6) * (t2 - t1));
        
    if (method == 0) {
    
        assert(0);
    
    }
    else {
    
        // robust tensor power method
        double* lambda = new double[K];
        double** v = new double*[K];
        for(int k = 0; k < K; k++)
            v[k] = new double[K];
            
        double residue = fast_collide_tensor_power_method(cs_T, K, K, L, T, fft_wrapper, ifft_wrapper, lambda, v, true);
        printf("residue = %lf\n", residue);
        
        tensor_lda_parameter_recovery(model, W, V, K, alpha0, lambda, v, mat_wrapper);
            
        delete[] lambda;
        for(int k = 0; k < K; k++)
            delete[] v[k];
        delete[] v;
    
    }
    
    for(int k = 0; k < K; k++)
        model->alpha[k] = alpha0 / K;
    model->alpha0 = alpha0;
    
    delete cs_T;
    delete cs_u;
    delete cs_w;
    delete cs_m1;
    delete hashes;
    delete[] coeff1;
    delete[] coeff2;
    delete[] sumwn;
    delete[] pw;

}
