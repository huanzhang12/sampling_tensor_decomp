#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "config.h"
#include "util.h"
#include "corpus.h"

Document::Document() {

	id = 0;
	idx = occs = NULL;
	num_items = num_words = 0;

}

Document::~Document() {

	if (idx) {
		delete[] idx;
		idx = NULL;
	}

	if (occs) {
		delete[] occs;
		occs = NULL;
	}

}

Corpus::Corpus(int V, int K) {

	this->V = V;
	this->K = K;

	num_data_files = 0;
	data_files = new char*[MAX_DOCUMENT_FILES];
	assert(data_files);

	num_docs = 0;
	docs = NULL;

}

Corpus::~Corpus() {

	clear();
	
	for(int i = 0; i < num_data_files; i++)
		delete[] data_files[i];
	delete[] data_files;

}

void Corpus::load(char* filename) {

	clear();

	printf("Loading corpus file %s\n", filename);

	// Please refer to readme.txt for a description of the binary file that stores a corpus
	
	FILE* fp = fopen(filename, "rb");
	assert(fp);

	fread(&num_docs, sizeof(int), 1, fp);
	docs = new Document[num_docs];
	for(int d = 0; d < num_docs; d++) {
		fread(&(docs[d].id), sizeof(long long), 1, fp);
		int num_items = 0;
		fread(&num_items, sizeof(int), 1, fp);
		docs[d].num_items = num_items;
		docs[d].idx = new int[num_items];
		docs[d].occs = new int[num_items];
		fread(docs[d].idx, sizeof(int), num_items, fp);
		fread(docs[d].occs, sizeof(int), num_items, fp);
		docs[d].num_words = 0;
		for(int i = 0; i < num_items; i++)
			docs[d].num_words += docs[d].occs[i];
	}

	fclose(fp);

}

void Corpus::save(char* filename) {

	printf("Saving corpus file %s\n", filename);

	FILE* fp = fopen(filename, "wb");
	assert(fp);

	fwrite(&num_docs, sizeof(int), 1, fp);
	for(Document* doc = docs; doc < docs + num_docs; doc ++) {
		fwrite(&(doc->id), sizeof(long long), 1, fp);
		fwrite(&(doc->num_items), sizeof(int), 1, fp);
		fwrite(doc->idx, sizeof(int), doc->num_items, fp);
		fwrite(doc->occs, sizeof(int), doc->num_items, fp);
	}

	fclose(fp);

}

void Corpus::append_df(char* filename) {

	data_files[num_data_files] = new char[MAX_FILENAME_LENGTH];
	assert(data_files[num_data_files]);
	memset(data_files[num_data_files], 0, MAX_FILENAME_LENGTH);
	strcpy(data_files[num_data_files], filename);
	
	printf("data_files[%d] = %s\n", num_data_files, data_files[num_data_files]);
	
	num_data_files ++;

}

void Corpus::clear() {

	if (docs) {
		delete[] docs;
	}
	
	/*for(int idf = 0; idf  < num_data_files; idf ++)
		delete[] data_files[idf];
	delete[] data_files;*/

	num_docs = 0;
	docs = NULL;

}
