#ifndef LDA_H_
#define LDA_H_

#include "corpus.h"
#include "matlab_wrapper.h"

class LDA {

public:
	LDA();
	LDA(int V, int K);
	~LDA();

	virtual int sanity_check();

	virtual void init_model();
	virtual Document* synth_docs(int num_docs, int num_words, Matlab_wrapper*);

	virtual void load(char* filename);
	virtual void load_ascii(char* filename, int format); // format = 1: V x K; format = 2: K x V; format = 3: K x V, log(p)
	virtual void save(char* filename);
    
    virtual double compute_perword_likelihood(Matlab_wrapper*, Corpus* corpus = NULL, bool abridged = false);
    
    virtual void gibbs(Corpus* corpus, double eps = 1e-4, int maxiter = 100);
    virtual double compute_gibbs_likelihood(Corpus* corpus);

	virtual void clear();

	int V, K;
	Corpus* corpus;

	double *alpha, *beta;
	double alpha0;
	double **Phi; // k x V

};

#endif
