#ifndef FFT_WRAPPER_H_
#define FFT_WRAPPER_H_

#include "fftw3.h"

class FFT_wrapper {

public:
	int len;
	int sign;

	FFT_wrapper(int len, int sign) {

		assert(len > 0);
		this->len = len;
		this->sign = sign;

		in = new fftw_complex[len];
		out = new fftw_complex[len];
		p = fftw_plan_dft_1d(len, in, out, sign, FFTW_MEASURE);

	}

	~FFT_wrapper() {

		fftw_destroy_plan(p);
		delete[] in;
		delete[] out;

	}

	void fft(fftw_complex* a, fftw_complex* fa) {

		memcpy(in, a, sizeof(fftw_complex) * len);
		fftw_execute(p);
		memcpy(fa, out, sizeof(fftw_complex) * len);

		if (sign == FFTW_BACKWARD) {
			double scale = 1.0 / len;
			for(int i = 0; i < len; i++) {
				fa[i][0] *= scale;
				fa[i][1] *= scale;
			}
		}

	}

private:
	fftw_complex *in, *out;
	fftw_plan p;

};

#endif
