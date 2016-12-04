
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "engine.h"

#include "util.h"
#include "config.h"
#include "LDA.h"
#include "corpus.h"

LDA::LDA() {

	this->V = 0;
	this->K = 0;

	corpus = NULL;
	alpha = beta = NULL;
	Phi = NULL;
	alpha0 = 0;

}

LDA::LDA(int V, int K) {

	this->V = V;
	this->K = K;

	corpus = NULL;

	alpha = new double[K];
	beta = new double[V];
	Phi = new double*[K];
	for(int k = 0; k < K; k++) {
		Phi[k] = new double[V];
	}

	init_model();

}

LDA::~LDA() {

	clear();

}

void LDA::init_model() {

	for(int k = 0; k < K; k++)
		alpha[k] = DEFAULT_ALPHA;
	alpha0 = DEFAULT_ALPHA * K;
	for(int i = 0; i < V; i++)
		beta[i] = DEFAULT_BETA;

	for(int k = 0; k < K; k++) {
		//random Gaussian entries
        generate_uniform_sphere_point(V, Phi[k]);
        double sum = 0;
        for(int i = 0; i < V; i++) {
            Phi[k][i] = fabs(Phi[k][i]);
            sum += Phi[k][i];
        }
        double scale = 1.0 / sum;
        for(int i = 0; i < V; i++)
            Phi[k][i] *= scale;
	}

}

Document* LDA::synth_docs(int num_docs, int num_words, Matlab_wrapper* mat_wrapper) {

	assert(sanity_check());
    printf("synth docs, num_docs = %d, num_words = %d\n", num_docs, num_words);

	Document* docs = new Document[num_docs];
	assert(docs);

	double* h = new double[K];
	double* p = new double[V];
	int* cnt = new int[V];
	assert(h);
	assert(cnt);
	
	puts("alpha:");
	for(int k = 0; k < K; k++)
	    printf("%lf ", alpha[k]);
	puts("");

	int thresh = (int)(0.01 * num_docs);
	int cnt_doc = 0;
	
	for(int d = 0; d < num_docs; d++) {

		if (cnt_doc++ > thresh) {putchar('.'); fflush(stdout); cnt_doc = 0;}

		docs[d].id = d;
		docs[d].num_words = num_words;
        
		mat_wrapper->drchrnd(K, alpha, h);
        
		for(int i = 0; i < V; i++) {
			p[i] = 0;
			for(int k = 0; k < K; k++)
				p[i] += h[k] * Phi[k][i];
		}
        
		mat_wrapper->mnrnd(num_words, p, V, cnt);

		docs[d].num_items = 0;
		for(int i = 0; i < V; i++)
			if (cnt[i] > 0) docs[d].num_items ++;

		docs[d].idx = new int[docs[d].num_items];
		docs[d].occs = new int[docs[d].num_items];


		int id = 0;
		for(int i = 0; i < V; i++)
			if (cnt[i] > 0) {
				docs[d].idx[id] = i;
				docs[d].occs[id++] = cnt[i];
			}


	}

	puts("");

	delete[] p;
	delete[] h;
	delete[] cnt;

	return docs;

}

int LDA::sanity_check() {

	if (K <= 0 || V <= 0) return 0;
    
	double sum = 0;
	for(int k = 0; k < K; k++) {
		if (alpha[k] <= 0) return 0;
		sum += alpha[k];
	}
	if (safe_compare(sum, alpha0)) return 0;

	for(int k = 0; k < K; k++) 
		if (beta[k] <= 0) return 0;

	for(int k = 0; k < K; k++) {
		double sum = 0;
		for(int i = 0; i < V; i++) {
			if (Phi[k][i] < 0 || Phi[k][i] > 1) return 0;
			sum += Phi[k][i];
		}
		if (safe_compare(sum, 1.0)) {
		    printf("sum = %lf\n", sum);
		    return 0;
		}
	}
	
	return 1;

}

void LDA::load(char* filename) {

	printf("Loading model file \"%s\"\n", filename);

	FILE* fp = fopen(filename, "rb");
	assert(fp);

	clear();

	fread(&V, sizeof(int), 1, fp);
	fread(&K, sizeof(int), 1, fp);

	alpha = new double[K];
	fread(alpha, sizeof(double), K, fp);
	alpha0 = 0;
	for(int k = 0; k < K; k++) alpha0 += alpha[k];

	beta = new double[V];
	fread(beta, sizeof(double), V, fp);

	Phi = new double*[K];
	for(int k = 0; k < K; k++) {
		Phi[k] = new double[V];
		fread(Phi[k], sizeof(double), V, fp);
	}

	sanity_check();

	fclose(fp);

}

void LDA::load_ascii(char* filename, int format) {

    printf("Loading ascii model file \"%s\"\n", filename);
    assert(format == 1 || format == 2 || format == 3);
    
    FILE* fp = fopen(filename, "r");
    assert(fp);

    clear();
    fscanf(fp, "%d %d\n", &V, &K);
    
    alpha = new double[K];
    for(int k = 0; k < K; k++)
        alpha[k] = DEFAULT_ALPHA;
    alpha0 = DEFAULT_ALPHA * K;
    beta = new double[V];
    for(int i = 0; i < V; i++)
        beta[i] = DEFAULT_BETA;
    
    Phi = new double*[K];
    for(int k = 0; k < K; k++) {
        Phi[k] = new double[V];
    }
    
    if (format == 1) {
        for(int i = 0; i < V; i++)
            for(int k = 0; k < K; k++)
                fscanf(fp, "%lf", &Phi[k][i]);
    }
    else if (format == 2) {
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < V; i++)
                fscanf(fp, "%lf", &Phi[k][i]);
			double sum = 0;
			for(int i = 0; i < V; i++) {
				if (Phi[k][i] < 0) Phi[k][i] = 0;
				sum += Phi[k][i];				
			}
			for(int i = 0; i < V; i++)
				Phi[k][i] /= sum;
		}
    }
	else if (format == 3) {
		double t = 0;
		for(int k = 0; k < K; k++) 
			for(int i = 0; i < V; i++) {
				fscanf(fp, "%lf", &t);
				Phi[k][i] = exp(t);
			}
	}
           
    sanity_check();
    
    fclose(fp);

}

void LDA::save(char* filename) {

	printf("Saving model file \"%s\"\n", filename);

	FILE* fp = fopen(filename, "wb");
	assert(fp);

	assert(sanity_check());

	fwrite(&V, sizeof(int), 1, fp);
	fwrite(&K, sizeof(int), 1, fp);

	fwrite(alpha, sizeof(double), K, fp);
	fwrite(beta, sizeof(double), V, fp);

	for(int k = 0; k < K; k++) {
		fwrite(Phi[k], sizeof(double), V, fp);
	}

	fclose(fp);

}

double LDA::compute_perword_likelihood(Matlab_wrapper* mat_wrapper, Corpus* corpus, bool abridged) {

    if (!corpus) corpus = this->corpus;
    double sumret = 0;
    int num_tot_docs = 0;
    
    for(int k = 0; k < K; k++)
        for(int i = 0; i < V; i++)
            this->Phi[k][i] += 1e-15;
    
    double* Aeq = new double[K];
    double* Beq = new double[1];
    for(int k = 0; k < K; k++)
        Aeq[k] = 1;
    Beq[0] = 1;
    double* C = new double[V*K];
    double* w = new double[V];
    for(int i = 0; i < V; i++)
        for(int k = 0; k < K; k++)
            C[IND2D(i,k,K)] = this->Phi[k][i];
    double* lb = new double[K];
    double* ub = new double[K];
    for(int k = 0; k < K; k++) {
        lb[k] = 0;
        ub[k] = 1;
    }
    double* pi = new double[K];
    
    Engine* ep = mat_wrapper->get_engine();
    mxArray* mxC = mat_wrapper->create_matrix(C, V, K, false);
    mxArray* mxlb = mat_wrapper->create_matrix(lb, K, 1, false);
    mxArray* mxub = mat_wrapper->create_matrix(ub, K, 1, false);
    mxArray* mxAeq = mat_wrapper->create_matrix(Aeq, 1, K, false);
    mxArray* mxBeq = mat_wrapper->create_matrix(Beq, 1, 1, false);
    engPutVariable(ep, "C", mxC);
    engPutVariable(ep, "lb", mxlb);
    engPutVariable(ep, "ub", mxub);
    engPutVariable(ep, "Aeq", mxAeq);
    engPutVariable(ep, "Beq", mxBeq);
    
    mxArray* mxw = mxCreateDoubleMatrix(V, 1, mxREAL);
    mxArray* mxx;
    
    for(int idf = 0; idf < corpus->num_data_files; idf ++) {
    
        corpus->load(corpus->data_files[idf]);
        int num_docs = corpus->num_docs;
        if (abridged) num_docs = (int)(0.1 * num_docs);
        int thresh = (int)(0.01 * num_docs);
        int cnt = 0;
        
        int p = 0;
        int perc = 0;
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
            if (cnt++ > thresh) {
                perc ++; cnt = 0;
                printf("%d%% completed, perplexity = %lf\n", perc, sumret / p);
            }
            int m = doc->num_words;
            double scale = 1.0 / m;
            memset(w, 0, sizeof(double) * V);
            for(int i = 0; i < doc->num_items; i++)
                w[doc->idx[i]] += doc->occs[i] * scale;
                
            memcpy(mxGetPr(mxw), w, sizeof(double) * V);
            engPutVariable(ep, "w", mxw);
            engEvalString(ep, "x = lsqlin(C, w, [], [], Aeq, Beq, lb, ub);");
            mxx = engGetVariable(ep, "x");
            memcpy(pi, mxGetPr(mxx), sizeof(double) * K);
            mxDestroyArray(mxx);

            double sum = 0;
            for(int k = 0; k < K; k++) {
                if (pi[k] < 1e-15) pi[k] = 1e-15;
                sum += pi[k];
            }
            for(int k = 0; k < K; k++)
                pi[k] /= sum;
            // compute perword likelihood
            for(int i = 0; i < doc->num_items; i++) {
                int v = doc->idx[i];
                double prob = 0;
                for(int k = 0; k < K; k++)
                    prob += pi[k] * Phi[k][v];
                sumret += scale * doc->occs[i] * log(prob);
            }
            p ++;
        }
        
        puts("");
        num_tot_docs += num_docs;
    
    }
    
    delete[] Aeq;
    delete[] Beq;
    delete[] C;
    delete[] w;
    delete[] lb;
    delete[] ub;
    delete[] pi;
    
    mxDestroyArray(mxC);
    mxDestroyArray(mxlb);
    mxDestroyArray(mxub);
    mxDestroyArray(mxAeq);            
    mxDestroyArray(mxBeq);
    mxDestroyArray(mxw);
    
    return sumret / num_tot_docs;

}

double LDA::compute_gibbs_likelihood(Corpus* corpus) {

    int num_docs = corpus->num_docs;
    double ret = 0;
    for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        int m = doc->num_words;
        // theta
        for(int k = 0; k < K; k++)
            ret += (alpha[k]-1) * log(doc->theta[k]);
        // z
        for(int i = 0; i < m; i++) {
            int v = doc->words[i], z = doc->topics[i];
            ret -= log(doc->theta[z]) + log(Phi[z][v]);
        }
    }
    
    return ret;

}

void LDA::gibbs(Corpus* corpus, double eps, int maxiter) {

    int num_docs = corpus->num_docs;
    printf("num_docs = %d\n", num_docs);
    assert(num_docs > 0);
    
    for(int k = 0; k < K; k++)
        alpha[k] = 50.0 / K;
    alpha0 = 50;
    for(int i = 0; i < V; i++)
        beta[i] = 0.1;
        
    for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc ++) {
        int m = doc->num_words;
        doc->theta = new double[K];
        doc->words = new int[m];
        doc->topics = new int[m];
        assert(doc->theta);
        assert(doc->words);
        assert(doc->topics);
        int p = 0;
        dirichlet_sampling(doc->theta, alpha, K);
        for(int i = 0; i < doc->num_items; i++)
            for(int j = 0; j < doc->occs[i]; j++) {
                doc->words[p] = doc->idx[i];
                doc->topics[p++] = rand() % K;
            }
        assert(p == m);
    }
    
    puts("Initialization completed.");
    
    double* p0 = new double[K];
    
    double lastprob = compute_gibbs_likelihood(corpus);
    for(int iter = 0; iter < maxiter; iter++) {
    
        printf("Iter %d: ", iter);
        fflush(stdout);
        
        // sample theta
        printf("theta ... "); fflush(stdout);
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc++) {
            int m = doc->num_words;
            for(int k = 0; k < K; k++)
                p0[k] = alpha[k];
            for(int i = 0; i < m; i++)
                p0[doc->topics[i]] += 1;
            dirichlet_sampling(doc->theta, p0, K);
        }
        
        // sample z
        printf("z ... "); fflush(stdout);
        for(Document* doc = corpus->docs; doc < corpus->docs + num_docs; doc++) {
            int m = doc->num_words;
            for(int i = 0; i < m; i++) {
                int v = doc->words[i];
                for(int k = 0; k < K; k++)
                    p0[k] = doc->theta[k] * Phi[k][v];
                doc->topics[i] = categorical_sampling(p0, K);
            }
        }
        
        printf("eval ... "); fflush(stdout);
        double thisprob = compute_gibbs_likelihood(corpus);
        printf("%lf, deltafit = %lf\n", thisprob, fabs((thisprob - lastprob) / lastprob));
        fflush(stdout);
        if (fabs((thisprob - lastprob) / lastprob) < eps) break;
        lastprob = thisprob;
    
    }
    
    delete[] p0;

}

void LDA::clear() {

	if (alpha) delete[] alpha;
	alpha = NULL;
	alpha0 = 0;
	if (beta) delete[] beta;
	beta = NULL;

	if (Phi) {
		for(int k = 0; k < K; k++)
			delete[] Phi[k];
		delete[] Phi;
	}
	Phi = NULL;

	V = 0;
	K = 0;

}
