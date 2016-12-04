#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>

#include "engine.h"

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
extern Hashes* debug_tensor_lda_hashes[3];
extern AsymCountSketch* debug_tensor_lda_cs;
extern CountSketch* debug_tensor_lda_sym_cs;
extern Hashes* debug_tensor_lda_sym_hashes;
#endif

typedef struct {

    int d_start, d_end;
    int B, b;
    FFT_wrapper* fft_wrapper;
    FFT_wrapper* ifft_wrapper;    

    Corpus* corpus;
    double alpha0;
    double* W;
    
    int v_start, v_end;
    double* M2;
    
    CountSketch* cs_T;
    CountSketch* f_cs_T;
    CountSketch* cs_m1;
    double* pw;
    int* num_words;
    double* coeff1;
    double* coeff2;
    double* sumwn;
    
    Tensor* tensor;
    
    int L, T;

} strct_workspace;

void* subprocess_whiten(void* para) {

    strct_workspace* data = reinterpret_cast<strct_workspace*>(para);
    
    size_t V = data->corpus->V;
    int K = data->corpus->K;
    Corpus* corpus = data->corpus;
    double* M2 = data->M2;
    int v_start = data->v_start;
    int v_end = data->v_end;
    
    printf("Start, pv = %d --> %d\n", v_start, v_end);
    
    long long* tot_num_docs = new long long;
    *tot_num_docs = 0;

    int pdoc = 0;
    int thresh = (int)(0.01 * corpus->num_docs);
	for(Document* doc = corpus->docs; doc < corpus->docs + corpus->num_docs; doc ++, pdoc ++) {
	
		if (doc->num_words < 2) continue;
		*tot_num_docs = *tot_num_docs + 1;
		double scale = 1.0 / ((double)(doc->num_words) * (doc->num_words - 1));
		int p_start = 0;
		while (p_start < doc->num_items && doc->idx[p_start] < v_start) p_start ++;
		int p_end = p_start;
		while (p_end < doc->num_items && doc->idx[p_end] <= v_end) p_end ++;

		for(int i = p_start; i < p_end; i++)
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
		
    printf("Completed, v_start = %d\n", v_start);
    return reinterpret_cast<void*>(tot_num_docs);		
		
}

void multithread_slow_whiten(int n_threads, Corpus* corpus, double alpha0, double* M1, double* W, Matlab_wrapper* mat_wrapper) {

	size_t V = corpus->V;
	int K = corpus->K;
	double scale = 0;
	
	printf("Computing word co-occurrence ...\n");
	
    pthread_t* threads = new pthread_t[n_threads];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	double* M2 = new double[V*V];
	assert(M2);
	memset(M2, 0, sizeof(double) * V*V);	
	printf("Memory allocation finished.\n");
	
    strct_workspace* workspaces = new strct_workspace[n_threads];	
    int v_batch = V / n_threads;
    for(int p = 0; p < n_threads; p++) {
        workspaces[p].corpus = corpus;
        workspaces[p].M2 = M2;
        workspaces[p].v_start = p * v_batch;
        workspaces[p].v_end = (p+1) * v_batch - 1;
    }
    workspaces[n_threads-1].v_end = V-1;

	// empirical 2-nd order moment

	printf("Computing empirical 2-nd order moment ...\n");
	
	long long tot_num_docs = 0;
	for(int idf = 0; idf < corpus->num_data_files; idf++) {

		corpus->load(corpus->data_files[idf]);

        struct timeval t1, t2;
        gettimeofday(&t1, NULL);
        
        for(int p = 0; p < n_threads; p++) {
            int rc = pthread_create(&threads[p], &attr, subprocess_whiten, reinterpret_cast<void*>(&workspaces[p]));
            assert(!rc);       
        }
        for(int p = 0; p < n_threads; p++) {
            void* status;
            int rc = pthread_join(threads[p], &status);
            assert(!rc);
            long long* ret = reinterpret_cast<long long*>(status);
            tot_num_docs = *ret;
            delete ret;
        }
        
        gettimeofday(&t2, NULL);
        printf("Elapsed time = %lf\n", (double)(t2.tv_sec - t1.tv_sec));

	}	
	
	scale = 1.0 / tot_num_docs;
	for(int p = 0; p < SQR(V); p++)
		M2[p] *= scale;

	// sanity check
	double sum = 0;
	for(int p = 0; p < SQR(V); p++)
		sum += M2[p];
	printf("M2 sum = %lf, diff = %e\n", sum, fabs(sum - 1.0));
	assert(safe_compare(sum, 1.0, 1e-4) == 0);

	// add 1-st order correction
	scale = alpha0 / (alpha0 + 1);
	for(int i = 0; i < V; i++)
		for(int j = 0; j < V; j++)
			M2[IND2D(i,j,V)] -= scale * M1[i] * M1[j];

	// whitening: using Matlab svds
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
			
	delete[] M2;
	delete[] UL;
	delete[] S;

}

void* subprocess_build_cs_part1(void* para) {

    strct_workspace* data = reinterpret_cast<strct_workspace*>(para);
    
    size_t V = data->corpus->V;
    int K = data->corpus->K;
    int B = data->B;
    int b = data->b;
    double* pw = data->pw;
    int* num_words = data->num_words;
    double* W = data->W;
    double alpha0 = data->alpha0;
    Corpus* corpus= data->corpus;
    int num_docs = corpus->num_docs;
    CountSketch* cs_T = data->cs_T;
    CountSketch* cs_m1 = data->cs_m1;
    int d_start = data->d_start;
    int d_end = data->d_end;
    
    fftw_complex* ret = new fftw_complex[(data->d_end - data->d_start + 1) * POWER2(b)];
    for(int d = d_start; d <= d_end; d++) {
        memcpy(ret + (d-d_start) * POWER2(b), cs_T->cs[d], sizeof(fftw_complex) * POWER2(b));
    }
    
    printf("Thread with d: %d --> %d\n", data->d_start, data->d_end);
    
    double* pw_base = new double[K];
    for(int k = 0; k < K; k++)
        pw_base[k] = (double)rand() / RAND_MAX - 0.5;
    
    fftw_complex* cs_u = new fftw_complex[POWER2(b)];
    fftw_complex t;
    int pdoc = 0;
    for(pdoc = 0; pdoc < num_docs; pdoc ++) {
    
        int m = num_words[pdoc];
        if (m < 3) continue;
        double scale1 = 1.0 / ((double)m * (m-1) * (m-2));
        double scale2 = alpha0 / ((alpha0+2) * m * (m-1));         

        double* pw_base = pw + pdoc * K;
        
        for(int d = data->d_start; d <= data->d_end; d++) {
            memset(cs_u, 0, sizeof(fftw_complex) * POWER2(b));
            fftw_complex* ret_base = ret + (d-data->d_start) * POWER2(b);
            for(int i = 0; i < K; i++) {
                int ind = cs_T->h->H[d][i];
                int omega = cs_T->h->Sigma[d][i];
                double value = pw_base[i];
                cs_u[ind][0] += value * Hashes::Omega[omega][0];
                cs_u[ind][1] += value * Hashes::Omega[omega][1];
            }            
            data->fft_wrapper->fft(cs_u, cs_u);
            for(int i = 0; i < POWER2(b); i++) {
                // u \tensor u \tensor u
                complex_assign(cs_u[i], t);
                complex_mult(t, cs_u[i], t);
                complex_mult(t, cs_u[i], t);
                ret_base[i][0] += scale1 * t[0];
                ret_base[i][1] += scale1 * t[1];
                // u \tensor u \tensor M1
                complex_assign(cs_u[i], t);
                complex_mult(t, cs_u[i], t);
                complex_mult(t, cs_m1->cs[d][i], t);
                ret_base[i][0] -= scale2 * 3 * t[0];
                ret_base[i][1] -= scale2 * 3 * t[1];
            }
        }
    
    }
    assert(pdoc == num_docs);    
    
    delete cs_u;
    
    printf("Completed, d_start = %d\n", data->d_start);
    return reinterpret_cast<void*>(ret);

}

void* subprocess_build_cs_part2(void* para) {

    strct_workspace* data = reinterpret_cast<strct_workspace*>(para);
    
    size_t V = data->corpus->V;
    int K = data->corpus->K;
    int B = data->B;
    int b = data->b;
    double* pw = data->pw;
    double* W = data->W;
    double alpha0 = data->alpha0;
    double* coeff1 = data->coeff1;
    double* coeff2 = data->coeff2;
    double* sumwn = data->sumwn;
    Corpus* corpus= data->corpus;
    int num_docs = corpus->num_docs;
    CountSketch* cs_T = data->cs_T;
    CountSketch* cs_m1 = data->cs_m1;
    int d_start = data->d_start;
    int d_end = data->d_end;    
    
    printf("Thread with d: %d --> %d\n", data->d_start, data->d_end);    
    
    fftw_complex* cs_w = new fftw_complex[POWER2(b)];
    fftw_complex* cs_u = new fftw_complex[POWER2(b)];
    fftw_complex* ret = new fftw_complex[(data->d_end - data->d_start + 1) * POWER2(b)];
    for(int d = d_start; d <= d_end; d++) {
        memcpy(ret + (d-d_start) * POWER2(b), cs_T->cs[d], sizeof(fftw_complex) * POWER2(b));
    }
    
    fftw_complex t;
    for(int i = 0; i < V; i++) {
        for(int d = data->d_start; d <= data->d_end; d++) {
            memset(cs_w, 0, sizeof(fftw_complex) * POWER2(b));
            memset(cs_u, 0, sizeof(fftw_complex) * POWER2(b));
            fftw_complex* ret_base = ret + (d - data->d_start) * POWER2(b);
            double* W_base = W + i * K;
            double* sumwn_base = sumwn + i * K;
            for(int j = 0; j < K; j++) {
                int ind = cs_T->h->H[d][j];
                int omega = cs_T->h->Sigma[d][j];
                cs_w[ind][0] += W_base[j] * Hashes::Omega[omega][0];
                cs_w[ind][1] += W_base[j] * Hashes::Omega[omega][1];
                cs_u[ind][0] += sumwn_base[j] * Hashes::Omega[omega][0];
                cs_u[ind][1] += sumwn_base[j] * Hashes::Omega[omega][1];
            }
            data->fft_wrapper->fft(cs_w, cs_w);
            data->fft_wrapper->fft(cs_u, cs_u);
            for(int j = 0; j < POWER2(b); j++) {
                // w \tensor w \tensor u
                complex_assign(cs_w[j], t);
                complex_mult(t, cs_w[j], t);
                complex_mult(t, cs_u[j], t);
                ret_base[j][0] -= 3 * t[0];
                ret_base[j][1] -= 3 * t[1];
            }
            for(int j = 0; j < POWER2(b); j++) {
                // w \tensor w \tensor w
                complex_assign(cs_w[j], t);
                complex_mult(t, cs_w[j], t);
                complex_mult(t, cs_w[j], t);
                ret_base[j][0] += coeff1[i] * t[0];
                ret_base[j][1] += coeff1[i] * t[1];
            }
            for(int j = 0; j < POWER2(b); j++) {
                // w \tensor w \tensor m1
                complex_assign(cs_w[j], t);
                complex_mult(t, cs_w[j], t);
                complex_mult(t, cs_m1->cs[d][j], t);
                ret_base[j][0] += coeff2[i] * 3 * t[0];
                ret_base[j][1] += coeff2[i] * 3 * t[1];
            }
        }
    }
    
    delete cs_w;
    delete cs_u;
    
    printf("Completed, d_start = %d\n", data->d_start);
    return reinterpret_cast<void*>(ret);

}

void* subprocess_rbp(void* para) {

    strct_workspace* data = reinterpret_cast<strct_workspace*>(para);
    
    size_t V = data->corpus->V;
    int K = data->corpus->K;
    int B = data->B;
    int b = data->b;
    double* pw = data->pw;
    double* W = data->W;
    double alpha0 = data->alpha0;
    Corpus* corpus= data->corpus;
    int num_docs = corpus->num_docs;
    CountSketch* cs_T = data->cs_T;
    CountSketch* f_cs_T = data->f_cs_T;
    int L = data->L;
    int T = data->T;
    
    CountSketch* cs_u = new CountSketch(cs_T->h);
    CountSketch* cs_uu = new CountSketch(cs_T->h);
    double* u = new double[K];
    double* v = new double[K+1];
    double max_value = 1e-100;

    for(int tau = 0; tau < data->L; tau++) {
    
        // Draw u randomly from the unit sphere and create its FFT count sketch
		generate_uniform_sphere_point(K, u);
		cs_u->set_vector(u, K, 1);
		cs_uu->set_vector(u, K, 2);

		for(int t = 0; t < T; t++) {
			fast_collide_TIuu(cs_T, f_cs_T, cs_u, cs_uu, u, K, data->fft_wrapper, data->ifft_wrapper, u);
			double norm = 0;
			for(int i = 0; i < K; i++)
				norm += SQR(u[i]);
				double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < K; i++)
				u[i] *= scale;
            cs_u->set_vector(u, K, 1);
            cs_uu->set_vector(u, K, 2);
		}

		// compute T(uuu) and update v[k]
		double value = fast_collide_Tuuu(cs_T, f_cs_T, cs_u, cs_uu, u, K, data->fft_wrapper, data->ifft_wrapper);
		if (value > max_value) {
		    max_value = value;
		    memcpy(v, u, sizeof(double) * K);
		}  
    
    }
    
    delete cs_u;
    delete cs_uu;
    delete[] u;
    
    v[K] = max_value;
    return reinterpret_cast<void*>(v);    

}

void* subprocess_rbp_naive(void* para) {

    strct_workspace* data = reinterpret_cast<strct_workspace*>(para);
    
    size_t V = data->corpus->V;
    int K = data->corpus->K;
    int L = data->L;
    int T = data->T;
    Tensor* tensor = data->tensor;
    
    double* u = new double[K];
    double* w = new double[K];
    double* v = new double[K+1];
    double max_value = 1e-100;

    for(int tau = 0; tau < data->L; tau++) {
    
        // Draw u randomly from the unit sphere and create its FFT count sketch
		generate_uniform_sphere_point(K, u);

		for(int t = 0; t < T; t++) {
		    tensor->TIuu(u, w, true);
		    memcpy(u, w, sizeof(double) * K);
			double norm = 0;
			for(int i = 0; i < K; i++)
				norm += SQR(u[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < K; i++)
				u[i] *= scale;
		}

		// compute T(uuu) and update v[k]
		double value = tensor->Tuuu(u, true);
		if (value > max_value) {
		    max_value = value;
		    memcpy(v, u, sizeof(double) * K);
		}  
    
    }
    
    delete[] u;
    delete[] w;
    
    v[K] = max_value;
    return reinterpret_cast<void*>(v);    

}

void multithread_fast_whiten_tensor_lda(int method, bool is_brute_force, int n_threads, Corpus* corpus, double alpha0, double* W, LDA* model, int B, int b, Matlab_wrapper* mat_wrapper, int T, int L, double* times) {

    size_t V = corpus->V;
    int K = corpus->K;
    double scale = 0, scale1 = 0, scale2 = 0;
    assert(T > 0 && L > 0);
    assert(method == 0 || method == 1);
    printf("fast multithread whitened tensor LDA: n_threads = %d, B = %d, b = %d, T = %d, L = %d\n", n_threads, B, b, T, L);

    FFT_wrapper** fft_wrappers = new FFT_wrapper*[n_threads];
    FFT_wrapper** ifft_wrappers = new FFT_wrapper*[n_threads];
    for(int p = 0; p < n_threads; p++) {
        fft_wrappers[p] = new FFT_wrapper(POWER2(b), FFTW_FORWARD);
        ifft_wrappers[p] = new FFT_wrapper(POWER2(b), FFTW_BACKWARD);
    }
    strct_workspace* workspaces = new strct_workspace[n_threads];
    assert(B % n_threads == 0);
    int d_per_thread = B / n_threads;
    for(int p = 0; p < n_threads; p++) {
        workspaces[p].B = B;
        workspaces[p].b = b;
        workspaces[p].d_start = p * d_per_thread;
        workspaces[p].d_end = (p+1) * d_per_thread - 1;
        workspaces[p].fft_wrapper = fft_wrappers[p];
        workspaces[p].ifft_wrapper = ifft_wrappers[p];
        workspaces[p].corpus = corpus;
        workspaces[p].alpha0 = alpha0;
        workspaces[p].W = W;
        workspaces[p].L = MAX(L / n_threads, 1);
        workspaces[p].T = T;
    }
    
    pthread_t* threads = new pthread_t[n_threads];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    struct timeval t1, t2;
    
    long long tot_num_docs = 0;
    
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
    CountSketch* f_cs_T = new CountSketch(hashes);
    CountSketch* cs_m1 = new CountSketch(hashes);
    cs_m1->set_vector(M1W, K, 1);
    cs_m1->fft(fft_wrappers[0]);
    for(int p = 0; p < n_threads; p++) {
        workspaces[p].cs_T = cs_T;
        workspaces[p].cs_m1 = cs_m1;
    }
        
    double* coeff1 = new double[V]; // w_i \tensor w_i \tensor w_i
    double* coeff2 = new double[V]; // w_i \tensor w_i \tensor m_1
    double* sumwn = new double[V * K];
    memset(coeff1, 0, sizeof(double) * V);
    memset(coeff2, 0, sizeof(double) * V);
    memset(sumwn, 0, sizeof(double) * V*K);
    
    if (times) {
        times[0] = 0;
        times[1] = 0;
        times[2] = 0;
    }
    
    for(int idf = 0; idf < corpus->num_data_files; idf++) {
    
        corpus->load(corpus->data_files[idf]);
        int num_docs = corpus->num_docs;
        double *pw = new double[num_docs * K];
        assert(pw);
        int* num_words = new int[num_docs];
        for(int p = 0; p < n_threads; p++) {
            workspaces[p].pw = pw;        
            workspaces[p].num_words = num_words;
        }
        
        gettimeofday(&t1, NULL);
        puts("Part 0 ---");
        
        int pdoc = 0;
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++, pdoc ++) {
            
            int m = doc->num_words;
            num_words[pdoc] = m;
            if (m < 3) {tot_num_docs --; continue;}
            scale1 = 1.0 / ((double)m * (m-1) * (m-2));
            scale2 = alpha0 / ((alpha0+2) * m * (m-1));            
            
            // compute pw
            double* pw_base = pw + pdoc * K;
            memset(pw_base, 0, sizeof(double) * K);
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                for(int k = 0; k < K; k++)
                    pw_base[k] += c * W[IND2D(p,k,K)];
            }            
            
            // Update coeff1, coeff2 and sumwn
            for(int i = 0; i < doc->num_items; i++) {
                int p = doc->idx[i], c = doc->occs[i];
                coeff1[p] += scale1 * 2 * c;
                coeff2[p] += scale2 * c;
                for(int k = 0; k < K; k++)
                    sumwn[IND2D(p,k,K)] += scale1 * c * pw_base[k];
            }            
            
        }
        assert(pdoc == num_docs);
        
        gettimeofday(&t2, NULL);
        printf("Elapsed time = %lf\n", (double)(t2.tv_sec - t1.tv_sec)); 
        if (times) times[0] += (double)(t2.tv_sec - t1.tv_sec);        
        
        gettimeofday(&t1, NULL);
        puts("Part 1 ---"); 
        
        for(int p = 0; p < n_threads; p++) {
            int rc = pthread_create(&threads[p], &attr, subprocess_build_cs_part1, reinterpret_cast<void*>(&workspaces[p]));
            assert(!rc);       
        }
        for(int p = 0; p < n_threads; p++) {
            void* status;
            int rc = pthread_join(threads[p], &status);
            assert(!rc);
            fftw_complex* ret = reinterpret_cast<fftw_complex*>(status);
            for(int d = workspaces[p].d_start; d <= workspaces[p].d_end; d++) {
                memcpy(cs_T->cs[d], ret + (d-workspaces[p].d_start) * POWER2(b), sizeof(fftw_complex) * POWER2(b));
            }
            delete[] ret;
        }
        
        gettimeofday(&t2, NULL);
        printf("Elapsed time = %lf\n", (double)(t2.tv_sec - t1.tv_sec));
        if (times) times[1] += (double)(t2.tv_sec - t1.tv_sec);
        
        tot_num_docs += num_docs;
        delete[] pw;
        delete[] num_words;
    
    } 
    
    
    gettimeofday(&t1, NULL);
    puts("Part 2 ---");
    
    for(int p = 0; p < n_threads; p++) {
        workspaces[p].coeff1 = coeff1;
        workspaces[p].coeff2 = coeff2;
        workspaces[p].sumwn = sumwn;
        int rc = pthread_create(&threads[p], &attr, subprocess_build_cs_part2, reinterpret_cast<void*>(&workspaces[p]));
        assert(!rc);
    }
    for(int p = 0; p < n_threads; p++) {
        void* status;
        int rc = pthread_join(threads[p], &status);
        assert(!rc);
        fftw_complex* ret = reinterpret_cast<fftw_complex*>(status);
        for(int d = workspaces[p].d_start; d <= workspaces[p].d_end; d++) {
            memcpy(cs_T->cs[d], ret + (d-workspaces[p].d_start) * POWER2(b), sizeof(fftw_complex) * POWER2(b));
        }
        delete[] ret;
    }
    
    gettimeofday(&t2, NULL);
    printf("Elapsed time = %lf\n", (double)(t2.tv_sec - t1.tv_sec));
    if (times) times[2] += (double)(t2.tv_sec - t1.tv_sec);
    
    // Normalize
    cs_T->fft(ifft_wrappers[0]);
    scale = 1.0 / tot_num_docs;
    for(int d = 0; d < B; d++)
        for(int i = 0; i < POWER2(b); i++) {
            cs_T->cs[d][i][0] *= scale;
            cs_T->cs[d][i][1] *= scale;
        }
        
    // Part 3
    scale = 2 * SQR(alpha0) / ((alpha0+1) * (alpha0+2));
    cs_T->add_rank_one_tensor(scale, M1W, K, fft_wrappers[0], ifft_wrappers[0], false);    
    
    #ifdef DEBUG_MODE_
    for(int d = 0; d < B; d++)
        for(int i = 0; i < POWER2(b); i++) {
          //  printf("(%lf %lf), (%lf %lf)\n", cs_T->cs[d][i][0], cs_T->cs[d][i][1], debug_tensor_lda_sym_cs->cs[d][i][0], debug_tensor_lda_sym_cs->cs[d][i][1]); 
            assert(safe_compare(cs_T->cs[d][i][0], debug_tensor_lda_sym_cs->cs[d][i][0]) == 0);
            assert(safe_compare(cs_T->cs[d][i][1], debug_tensor_lda_sym_cs->cs[d][i][1]) == 0);
        }
    return;
    #endif    
    
    if (method == 1 && !is_brute_force) {
    
    gettimeofday(&t1, NULL);
    puts("Robust tensor power method, with count sketch");
    
    double* lambda = new double[K];
    double** v = new double*[K];
    for(int k = 0; k < K; k++)
        v[k] = new double[K];
        
    double t_fnorm = fast_sqr_fnorm(cs_T);
    printf("Before deflation, norm = %lf\n", t_fnorm);
    for(int k = 0; k < K; k++) {
    
        printf("Round %d ...\n", k);
        f_cs_T->copy_from(cs_T);
        f_cs_T->fft(fft_wrappers[0]);
        
        for(int p = 0; p < n_threads; p++) {
            workspaces[p].cs_T = cs_T;
            workspaces[p].f_cs_T = f_cs_T;
            int rc = pthread_create(&threads[p], &attr, subprocess_rbp, reinterpret_cast<void*>(&workspaces[p]));
            assert(!rc);
        }
        
        lambda[k] = -1e100;
        for(int p = 0; p < n_threads; p++) {
            void* status;
            int rc = pthread_join(threads[p], &status);
            assert(!rc);
            strct_workspace* data = reinterpret_cast<strct_workspace*>(status);
            double* ret = reinterpret_cast<double*>(status);
            if (ret[K] > lambda[k]) {
                lambda[k] = ret[K];
                memcpy(v[k], ret, sizeof(double) * K);
            }
            delete[] ret;
        }
        
	cs_T->add_rank_one_tensor(-lambda[k], v[k], K, fft_wrappers[0], ifft_wrappers[0], false);
	printf("%lf\n", fast_sqr_fnorm(cs_T) / t_fnorm);
    
    }
    
    double residue = fast_sqr_fnorm(cs_T);
    printf("residue = %lf\n", residue / t_fnorm);
    
    gettimeofday(&t2, NULL);
    printf("Elapsed time = %lf\n", (double)(t2.tv_sec - t1.tv_sec));
    if (times) times[3] += (double)(t2.tv_sec - t1.tv_sec);
    
    tensor_lda_parameter_recovery(model, W, V, K, alpha0, lambda, v, mat_wrapper);
    model->alpha0 = alpha0;
    for(int k = 0; k < K; k++)
        model->alpha[k] = alpha0 / K;
            
    for(int k = 0; k < K; k++)
       delete[] v[k];
    delete[] v;
    delete[] lambda;
    
    }
    
    else if (method == 1 && is_brute_force) {
    
        gettimeofday(&t1, NULL);
        puts("Robust tensor power method, without count sketch");           
        
        double* lambda = new double[K];
        double** v = new double*[K];
        for(int k = 0; k < K; k++)
            v[k] = new double[K];    
        int* inds = new int[3];    
            
        Tensor* tensor = new Tensor(K, TENSOR_STORE_TYPE_DENSE);
        double scale1 = 1.0 / 6;
        double scale2 = 1.0 / 3;
        double scale3 = 1.0;
        for(int i = 0; i < K; i++)
            for(int j = i; j < K; j++)
                for(int k = j; k < K; k++) {
                    inds[0] = i, inds[1] = j, inds[2] = k;
                    double value = cs_T->read_entry(3, inds);
                    if (i == j && j == k)
                        value *= scale3;
                    else if (i == j || i == k || j == k)
                        value *= scale2;
                    else
                        value *= scale1;
                    tensor->A[IND3D(i,j,k,K)] = value;
                    tensor->A[IND3D(i,k,j,K)] = value;
                    tensor->A[IND3D(j,i,k,K)] = value;
                    tensor->A[IND3D(j,k,i,K)] = value;
                    tensor->A[IND3D(k,i,j,K)] = value;
                    tensor->A[IND3D(k,j,i,K)] = value;                                                                                                    
                }
                
        double t_fnorm = tensor->sqr_fnorm(true);
        printf("Before deflation, norm = %lf, sketched norm = %lf\n", t_fnorm, fast_sqr_fnorm(cs_T));
        tensor->save("LDA_wiki_tensor.dat");
        for(int k = 0; k < K; k++) {
        
            printf("Round %d ...\n", k);
            
            for(int p = 0; p < n_threads; p++) {
                workspaces[p].tensor = tensor;
                int rc = pthread_create(&threads[p], &attr, subprocess_rbp_naive, reinterpret_cast<void*>(&workspaces[p]));
                assert(!rc);
            }
            
            lambda[k] = -1e100;
            for(int p = 0; p < n_threads; p++) {
                void* status;
                int rc = pthread_join(threads[p], &status);
                assert(!rc);
                strct_workspace* data = reinterpret_cast<strct_workspace*>(status);
                double* ret = reinterpret_cast<double*>(status);
                if (ret[K] > lambda[k]) {
                    lambda[k] = ret[K];
                    memcpy(v[k], ret, sizeof(double) * K);
                }
                delete[] ret;
            }
            
            for(int i1 = 0; i1 < K; i1++)
                for(int i2 = 0; i2 < K; i2++)
                    for(int i3 = 0; i3 < K; i3++)
                        tensor->A[IND3D(i1,i2,i3,K)] -= lambda[k] * v[k][i1] * v[k][i2] * v[k][i3];                        
            printf("%lf\n", tensor->sqr_fnorm() / t_fnorm);
        
        }        
        
        double residue = tensor->sqr_fnorm(true);
        printf("after deflation, residue = %lf\n", residue);
        printf("residue = %lf\n", residue / t_fnorm);        
        
        gettimeofday(&t2, NULL);
        printf("Elapsed time = %lf\n", (double)(t2.tv_sec - t1.tv_sec));
        if (times) times[3] += (double)(t2.tv_sec - t1.tv_sec);        
        
        tensor_lda_parameter_recovery(model, W, V, K, alpha0, lambda, v, mat_wrapper);
        model->alpha0 = alpha0;
        for(int k = 0; k < K; k++)
            model->alpha[k] = alpha0 / K;
            
        for(int k = 0; k < K; k++)
           delete[] v[k];
        delete[] v;
        delete[] lambda;             
        delete tensor;
        delete[] inds;
    
    }
    
    else if (method == 1 && !is_brute_force) {
    
        puts("Start ALS, with count sketch ...");
        
        double* lambda = new double[K];
        double* AA = new double[K*K];
        double* BB = new double[K*K];
        double* CC = new double[K*K];
        for(int k = 0; k < K; k++) {
            generate_uniform_sphere_point(K, AA + k * K);
            memcpy(BB + k * K, AA + k * K, sizeof(double) * K);
            memcpy(CC + k * K, AA + k * K, sizeof(double) * K);
        }
        
        
    
    }
    
    for(int p = 0; p < n_threads; p++) {
        delete fft_wrappers[p];
        delete ifft_wrappers[p];
    }        
    delete[] fft_wrappers;
    delete[] ifft_wrappers;
    delete[] workspaces;
    delete[] M1;
    delete[] M1W;
    delete[] threads;
    delete[] coeff1;
    delete[] coeff2;
    delete[] sumwn;
    delete hashes;
    delete cs_T;
    delete f_cs_T;
    delete cs_m1;

}
