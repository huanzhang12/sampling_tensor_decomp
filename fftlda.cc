#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#include "config.h"
#include "util.h"
#include "hash.h"
#include "corpus.h"
#include "LDA.h"
#include "matlab_wrapper.h"
#include "fft_wrapper.h"
#include "tensor_lda.h"
#include "fast_tensor_power_method.h"

Matlab_wrapper* mat_wrapper;
FFT_wrapper *fft_wrapper, *ifft_wrapper;

int num_threads = 24;

int V = 10000;
int K = 100;

int L = 30;
int T = 20;

int B = 24;
int b = 10;

double alpha0 = 1.0;

double times[6];

clock_t t1, t2;

Corpus* corpus;
LDA *model, *ref_model;

void do_slow_tensor_lda(char* data_file, char* model_file) {

	corpus = new Corpus(V, K);
	corpus->append_df(data_file);
	model = new LDA(V, K);
	
	double* M1 = new double[V];
	double* W = new double[V*K];
	double* A2WW = new double[K*K];
	assert(M1 && W && A2WW);
	
	t1 = clock();
	compute_word_frequency(corpus, M1);
	slow_whiten(corpus, alpha0, M1, W, A2WW, mat_wrapper);
	slow_tensor_lda(corpus, alpha0, W, A2WW, model, mat_wrapper, T, L);
	t2 = clock();
	printf("Elapsed time = %lf\n", 1e-6 * (t2 - t1));
	
	printf("model_file = %s\n", model_file);
	model->save(model_file);

  //  double perp = model->compute_perword_likelihood(mat_wrapper, corpus, true);
   // printf("Perplexity: %lf\n", perp);
    
	delete[] M1;
	delete[] W;
	delete[] A2WW;
	
	delete corpus;
	delete model;

}

/*void do_fast_tensor_lda(char* data_file, char* model_file) {

	corpus = new Corpus(V, K);
	corpus->append_df(data_file);
	model = new LDA(V, K);
	
	double* M1 = new double[V];
	double* W = new double[V*K];
	double* A2WW = new double[K*K];
	assert(M1 && W && A2WW);
	
	compute_word_frequency(corpus, M1);
	slow_whiten(corpus, alpha0, M1, W, A2WW, mat_wrapper);
    
    Hashes *h = new Hashes(B, b, V, 6);
    //fast_tensor_lda(corpus, alpha0, W, model, h, mat_wrapper, fft_wrapper, ifft_wrapper, T);
    model->load("synth.model");
    
    //slow_als_lda(corpus, alpha0, W, model, h, mat_wrapper, fft_wrapper, ifft_wrapper, T);
    //fast_tensor_lda(corpus, alpha0, W, model, B, b, mat_wrapper, fft_wrapper, ifft_wrapper, T);
    
    
	puts("before saving model");
	printf("model_file = %s\n", model_file);
	model->save(model_file);
	puts("after saving model");
	
	delete[] M1;
	delete[] W;
	delete[] A2WW;
	
	delete corpus;
	delete model;    

}*/

void do_fast_whiten_tensor_lda(char* data_file, char* model_file, int method, bool is_brute_force) {

	corpus = new Corpus(V, K);
	corpus->append_df(data_file);
	model = new LDA(V, K);
	
	double* M1 = new double[V];
	double* W = new double[V*K];
	double* A2WW = new double[K*K];
	assert(M1 && W && A2WW);
	
	struct timeval t1, t2, t3;
	
    gettimeofday(&t1, NULL);
	compute_word_frequency(corpus, M1);
	
	multithread_slow_whiten(num_threads, corpus, alpha0, M1, W, mat_wrapper);
	
	#ifdef DEBUG_MODE_
	double* W_ref = new double[V*K];
	slow_whiten(corpus, alpha0, M1, W_ref, A2WW, mat_wrapper);
	for(int i = 0; i < V*K; i++) {
	    printf("%lf %lf\n", W[i], W_ref[i]);
	    assert(safe_compare(fabs(W[i]), fabs(W_ref[i])) == 0);
	}
	#endif
	
	gettimeofday(&t3, NULL);
	times[4] = (double)(t3.tv_sec - t1.tv_sec);
	printf("whitening time: %lf\n", times[4]);
	
    multithread_fast_whiten_tensor_lda(method, is_brute_force, num_threads, corpus, alpha0, W, model, B, b, mat_wrapper, T, L, times);

    gettimeofday(&t2, NULL);
    printf("Elapsed time = %lf\n", (double)(t2.tv_sec - t1.tv_sec));
    times[5] = (double)(t2.tv_sec - t1.tv_sec);
    
    FILE* fp = fopen("tmp.lda", "w");
    for(int k = 0; k < model->K; k++) {
        for(int i = 0; i < model->V; i++)
            fprintf(fp, "%lf ", model->Phi[k][i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    
	printf("model_file = %s\n", model_file);
	model->save(model_file);
    
 //   double perp = model->compute_perword_likelihood(mat_wrapper, corpus, true);
  //  printf("Perplexity: %lf\n", perp);
	
	delete[] M1;
	delete[] W;
	delete[] A2WW;
	
	delete corpus;
	delete model;    


}

void do_evaluate(char* model_file, char* ref_model_file) {

	model = new LDA(V, K);
	ref_model = new LDA(V, K);
	model->load(model_file);
	ref_model->load(ref_model_file);
	
	puts("alpha:");
	for(int k = 0; k < K; k++)
		printf("%lf ", model->alpha[k]);
	puts("");
	
	puts("===============================================");
	puts("Phi:");
	int* mates = new int[2*K];
	FILE* f = fopen("mwmatching.input", "w");
	assert(f);
	for(int k1 = 0; k1 < K; k1++)
		for(int k2 = 0; k2 < K; k2++) {
			double delta = compute_one_norm(ref_model->Phi[k1], model->Phi[k2], V);
			fprintf(f, "%d %d %lf\n", k1, K+k2, -delta);
		}
	fclose(f);
	
	system("python mwmatching.py");
	
	f = fopen("mwmatching.output", "r");
	assert(f);
	for(int i = 0; i < 2*K; i++)
		fscanf(f, "%d", &(mates[i])); 
	fclose(f);
	
	for(int k1 = 0; k1 < K; k1++) {
		int k2 = mates[k1] - K;
		printf("ref[%d] = %d: l1 norm = %lf\n", k1, k2, compute_one_norm(ref_model->Phi[k1], model->Phi[k2], V));
	}
	
	delete[] mates;
	delete model;
	delete ref_model;

}

double do_perp_evaluate(char* model_file, char* corpus_file, bool is_ascii_file = false) {

    model = new LDA(V, K);
    if (is_ascii_file)
        model->load_ascii(model_file, 2);
    else
        model->load(model_file);
    corpus = new Corpus(V, K);
    corpus->append_df(corpus_file);
    
    double perp = model->compute_perword_likelihood(mat_wrapper, corpus, false);
    printf("Perplexity: %lf\n", perp);
    
    delete model;
    delete corpus;
    
    return perp;

}

void do_gibbs(char* model_file, char* corpus_file, char* tassign_file) {

    model = new LDA(V, K);
    model->load(model_file);
    corpus = new Corpus(V, K);
    corpus->load(corpus_file);
    
    model->gibbs(corpus, 1e-4, 5);
    
    FILE* fp = fopen(tassign_file, "w");
    int num_docs = corpus->num_docs;
    int t = 0;
    for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        int m = doc->num_words;
        for(int i = 0; i < m-1; i++)
            fprintf(fp, "%d:%d ", doc->words[i], doc->topics[i]);
        if (m > 0) {
            fprintf(fp, "%d:%d\n", doc->words[m-1], doc->topics[m-1]);
        }
        else {
            t ++;
        }
    }
    printf("empty documents = %d\n", t);
    fclose(fp);

}

void get_int_from_env(const char * env_name, const char * var_name, int * val) {
	char * env = getenv(env_name);
	if (env != NULL) {
		*val = atoi(env);
		printf("Changing parameter %s to %d\n", var_name, *val);
		assert(val > 0);
	}
}

int main(int argc, char* argv[]) {

	srand(1991);
	get_int_from_env("CORPUS_V", "V", &V);
	get_int_from_env("CORPUS_K", "K", &K);
	get_int_from_env("POWER_L", "L", &L);
	get_int_from_env("POWER_T", "T", &T);

	mat_wrapper = new Matlab_wrapper();
	fft_wrapper = new FFT_wrapper(POWER2(b), FFTW_FORWARD);
	ifft_wrapper = new FFT_wrapper(POWER2(b), FFTW_BACKWARD);

	assert(argc > 1);


	if (strcmp(argv[1], "synth") == 0) {

		int num_docs = atoi(argv[2]);
		int num_words = atoi(argv[3]);

		corpus = new Corpus(V, K);
		model = new LDA(V, K);
		model->corpus = corpus;

		model->init_model();
		corpus->docs = model->synth_docs(num_docs, num_words, mat_wrapper);
		corpus->num_docs = num_docs;
	
		model->save("synth.model");
		corpus->save("synth.corpus");
		
		delete model;
		delete corpus;
		
	}
    
    else if (strcmp(argv[1], "fast_speclda") == 0) {

        do_fast_whiten_tensor_lda(argv[2], "fast.model", 1, false);
        if (argc > 3) {
            do_evaluate("fast.model", argv[3]); 
        }
    }
   else if (strcmp(argv[1], "brute_speclda") == 0) {

        do_fast_whiten_tensor_lda(argv[2], "brute.model", 1, true);
        if (argc > 3) {
            do_evaluate("brute.model", argv[3]);
        }
    }
   else if (strcmp(argv[1], "slow_speclda") == 0) {

        do_slow_tensor_lda(argv[2], "slow.model");
        if (argc > 3) {
            do_evaluate("slow.model", argv[3]);
        }
    }
    
    else if (strcmp(argv[1], "eval") == 0) {
    
        LDA* model = new LDA(V, K);
        double perp = do_perp_evaluate(argv[2], argv[3], true);
        printf("Perplexity: %lf\n", perp);
        
        FILE* f = fopen(argv[4], "a");
        fprintf(f, "%lf\n", perp);
        fclose(f);
    
    }
    
    else if (strcmp(argv[1], "gibbs") == 0) {
    
        do_gibbs(argv[2], argv[3], argv[4]);
    
    }

	else {
		assert(0);
	}

	return 0;

}
