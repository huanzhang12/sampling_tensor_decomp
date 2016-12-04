#ifndef CORPUS_H_
#define CORPUS_H_

class Document {

public:
	Document();
	~Document();

	long long id;
	int num_items; // #. of distinct words
	int num_words; // #. of words
	int *idx, *occs;
	
	// for gibbs use only
	double* theta = NULL;
	int* words = NULL; 
	int* topics = NULL;

};

class Corpus {

public:
	Corpus(int V, int K);
	~Corpus();

	void load(char* filename);
	void save(char* filename);
	
	void append_df(char*);

	void clear();

	int V, K; // vocabulary size && number of topics
	char** data_files; 
	int num_data_files;

	Document* docs;
	int num_docs;

};

#endif
