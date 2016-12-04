#ifndef HASH_C_
#define HASH_C_

#include "fftw3.h"

class Hashes {

public:
	static const fftw_complex* Omega;

	int** H;
	int** Sigma;
	int B, b, dim, K;

	Hashes(int, int, int, int);
	~Hashes();
    
    void to_asymmetric_hash();

	int sanity_check(int B, int b, int dim) {
		return (this->B == B && this->b == b && this->dim <= dim);
	}

private:
	unsigned int** C;

	void clear();
	static fftw_complex* init_omega();

};

#endif
