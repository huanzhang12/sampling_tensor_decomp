#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "fftw3.h"
#include "config.h"
#include "util.h"
#include "hash.h"

fftw_complex hash_omega_data[HASH_OMEGA_PERIOD];

fftw_complex* Hashes::init_omega() {

	for(int k = 0; k < HASH_OMEGA_PERIOD; k++) {
		hash_omega_data[k][0] = cos(2.0 * k * PI / HASH_OMEGA_PERIOD);
		hash_omega_data[k][1] = sin(2.0 * k * PI / HASH_OMEGA_PERIOD);
	}

	return hash_omega_data;

}

const fftw_complex* Hashes::Omega = Hashes::init_omega();

Hashes::Hashes(int B, int b, int dim, int K) {

	assert(1 <= B);
	assert(1 <= b && b <= MAX_LOG_HASH_LEN);
	assert(dim > 0);
	assert(K > 0);

	this->B = B;
	this->b = b;
	this->dim = dim;
	this->K = K;

	// build C and H
	H = new int*[B];
	C = new unsigned int*[B];
	for(int d = 0; d < B; d++) {
		C[d] = new unsigned int[K];
		for(int k = 0; k < K; k++)
			C[d][k] = (unsigned int)rand() & MASK2(b);
		H[d] = new int[dim];
		for(int i = 0; i < dim; i++) {
			unsigned int t = 1;
			unsigned int x = (unsigned int)(x+1);
			unsigned int code = 0;
			for(int k = 0; k < K; k++) {
				code += C[d][k] * t;
				t *= x;
			}
			H[d][i] = (int)(code & MASK2(b));
		}
	}

	// Build Sigma
	Sigma = new int*[B];
	for(int d = 0; d < B; d++) {
		Sigma[d] = new int[dim];
		for(int i = 0; i < dim; i++) {
			Sigma[d][i] = rand() & (HASH_OMEGA_PERIOD-1);	
		}
	}

}

Hashes::~Hashes() {
	clear();
}

void Hashes::clear() {

	for(int d = 0; d < B; d++) {
		delete[] H[d];
		delete[] Sigma[d];
		delete[] C[d];
	}

	delete[] H;
	delete[] Sigma;
	delete[] C;

}

void Hashes::to_asymmetric_hash() {

    for(int d = 0; d < B; d++)
        for(int i = 0; i < dim; i++)
            Sigma[d][i] = (Sigma[d][i] & 1)? 1 : -1;

}
