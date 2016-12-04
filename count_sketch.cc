#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "fftw3.h"
#include "config.h"
#include "util.h"
#include "tensor.h"
#include "hash.h"
#include "fft_wrapper.h"
#include "count_sketch.h"

CountSketch::CountSketch(Hashes* h) {

	this->h = h;
	this->B = h->B;
	this->b = h->b;
	this->src_data = NULL;

	cs = new fftw_complex*[B];
	for(int d = 0; d < B; d++) {
		cs[d] = new fftw_complex[POWER2(b)];
	}
	init();

}

CountSketch::~CountSketch() {

	for(int d = 0; d < B; d++)
		delete[] cs[d];
	delete[] cs;

}

void CountSketch::copy_from(CountSketch* src) {

	assert(B == src->B && b == src->b && h == src->h);

	this->src_data = src->src_data;
	for(int d = 0; d < B; d++)
		memcpy(this->cs[d], src->cs[d], sizeof(fftw_complex) * POWER2(b));

}

double CountSketch::read_entry(int order, int* inds) {

	double* values = new double[B];
	fftw_complex t;
	assert(values);
	
	for(int d = 0; d < B; d++) {
		int angle = 0;
		int p = 0;
		for(int i = 0; i < order; i++) {
			angle += h->Sigma[d][inds[i]];
			p += h->H[d][inds[i]];
		}
		angle = angle & (HASH_OMEGA_PERIOD - 1);
		p = p & MASK2(b);
		complex_mult_conj(cs[d][p], Hashes::Omega[angle], t);
		values[d] = t[0];
	}
	
	qsort(values, B, sizeof(double), compare_double);
	double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);	
	
	delete[] values;
	return ret;

}

void CountSketch::add_entry(int order, int* inds, double value) {

    for(int d = 0; d < B; d++) {
        int angle = 0;
        int p = 0;
        for(int i = 0; i < order; i++) {
            angle += h->Sigma[d][inds[i]];
            p += h->H[d][inds[i]];
        }
        angle = angle & (HASH_OMEGA_PERIOD - 1);
        p = p & MASK2(b);
        cs[d][p][0] += Hashes::Omega[angle][0] * value;
        cs[d][p][1] += Hashes::Omega[angle][1] * value;
    }

}

void CountSketch::set_tensor(Tensor* T, bool with_symmetric_adjustment) {

	assert(h->sanity_check(B, b, T->dim));
	src_data = T;

	init();
    
    if (T->store_type == TENSOR_STORE_TYPE_DENSE) {
        fftw_complex t;
        if (with_symmetric_adjustment) {
        for(int d = 0; d < B; d++) {
            for(int i = 0; i < T->dim; i++)
                for(int j = i; j < T->dim; j++)
                    for(int k = j; k < T->dim; k++) {
                        int ind = (h->H[d][i] + h->H[d][j] + h->H[d][k]) & MASK2(b);
                        int angle = (h->Sigma[d][i] + h->Sigma[d][j] + h->Sigma[d][k]) & (HASH_OMEGA_PERIOD-1);
                        double value = T->A[IND3D(i,j,k,T->dim)];
                        t[0] = Hashes::Omega[angle][0] * value;
                        t[1] = Hashes::Omega[angle][1] * value;
                        complex_add(cs[d][ind], t, cs[d][ind]);
                    }
        }
        }
        else {
            for(int d = 0; d < B; d++) {
                for(int i = 0; i < T->dim; i++)
                    for(int j = 0; j < T->dim; j++)
                        for(int k = 0; k < T->dim; k++) {
                           int ind = (h->H[d][i] + h->H[d][j] + h->H[d][k]) & MASK2(b);
                           int angle = (h->Sigma[d][i] + h->Sigma[d][j] + h->Sigma[d][k]) & (HASH_OMEGA_PERIOD-1);
                           double value = T->A[IND3D(i,j,k,T->dim)];
                           t[0] = Hashes::Omega[angle][0] * value;
                           t[1] = Hashes::Omega[angle][1] * value;
                           complex_add(cs[d][ind], t, cs[d][ind]);                        
                        }
            }
        }
    }
    else {
        assert(0);
    }

}

void CountSketch::set_vector(double* u, int len, int order, bool with_symmetric_adjustment) {

	assert(h->sanity_check(B, b, len));
	src_data = u;

	init();
	fftw_complex t;
	for(int d = 0; d < B; d++) {
		for(int i = 0; i < len; i++) {
			int ind = (h->H[d][i] * order) & MASK2(b);
			int angle = (h->Sigma[d][i] * order) & (HASH_OMEGA_PERIOD - 1);
			double value = 1;
			for(int l = 0; l < order; l++) value *= u[i];
			t[0] = Hashes::Omega[angle][0] * value;
			t[1] = Hashes::Omega[angle][1] * value;
			complex_add(cs[d][ind], t, cs[d][ind]);
		}
	}

}

// T(i,j,k) = lambda * u(i) * u(j) * u(k)
void CountSketch::add_rank_one_tensor(double lambda, double* u, int dim, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper, bool with_symmetric_adjustment) {

	assert(h->sanity_check(B, b, dim));

    if (!with_symmetric_adjustment) {
        fftw_complex t;
        fftw_complex* s_u = new fftw_complex[POWER2(b)];
        for(int d = 0; d < B; d++) {
            memset(s_u, 0, sizeof(fftw_complex) * POWER2(b));        
            for(int i = 0; i < dim; i++) {
                int ind = h->H[d][i];
                int angle = h->Sigma[d][i];
                s_u[ind][0] += Hashes::Omega[angle][0] * u[i];
                s_u[ind][1] += Hashes::Omega[angle][1] * u[i];                
            }
            fft_wrapper->fft(s_u, s_u);
            for(int i = 0; i < POWER2(b); i++) {
                complex_assign(s_u[i], t);
                complex_mult(t, s_u[i], t);
                complex_mult(t, s_u[i], t);
                complex_assign(t, s_u[i]);
            }
            ifft_wrapper->fft(s_u, s_u);
            for(int i = 0; i < POWER2(b); i++) {
                cs[d][i][0] += lambda * s_u[i][0];
                cs[d][i][1] += lambda * s_u[i][1];
            }
        }
        return;
    }

	double scale_a = 1.0 / 6;
	double scale_b = 3.0 / 6;
	double scale_c = 2.0 / 6;
	fftw_complex *s_u = new fftw_complex[POWER2(b)];
	fftw_complex *s_uu = new fftw_complex[POWER2(b)];
	fftw_complex *f_ab = new fftw_complex[POWER2(b)];
	fftw_complex t;
	
	// a and b
	for(int d = 0; d < B; d++) {

		memset(s_u, 0, sizeof(fftw_complex) * POWER2(b));
		for(int i = 0; i < dim; i++) {
			int ind = h->H[d][i];
			int angle = h->Sigma[d][i];
			s_u[ind][0] += Hashes::Omega[angle][0] * u[i];
			s_u[ind][1] += Hashes::Omega[angle][1] * u[i];
		}

		memset(s_uu, 0, sizeof(fftw_complex) * POWER2(b));
		for(int i = 0; i < dim; i++) {
			int ind = (h->H[d][i] << 1) & MASK2(b);
			int angle = (h->Sigma[d][i] << 1) & (HASH_OMEGA_PERIOD - 1);
			s_uu[ind][0] += Hashes::Omega[angle][0] * u[i] * u[i];
			s_uu[ind][1] += Hashes::Omega[angle][1] * u[i] * u[i];
		}

		fft_wrapper->fft(s_u, s_u);
		fft_wrapper->fft(s_uu, s_uu);

		// the B part
		for(int i = 0; i < POWER2(b); i++) {
			f_ab[i][0] = scale_b * (s_uu[i][0] * s_u[i][0] - s_uu[i][1] * s_u[i][1]);
			f_ab[i][1] = scale_b * (s_uu[i][0] * s_u[i][1] + s_uu[i][1] * s_u[i][0]);
		}

		// the A part
		for(int i = 0; i < POWER2(b); i++) {
			t[0] = s_u[i][0]; t[1] = s_u[i][1];
			complex_mult(s_u[i], t, s_u[i]);
			complex_mult(s_u[i], t, s_u[i]);
			f_ab[i][0] += scale_a * s_u[i][0];
			f_ab[i][1] += scale_a * s_u[i][1];
		}

		// adding f_ab back to current count sketch
		ifft_wrapper->fft(f_ab, f_ab);
		for(int i = 0; i < POWER2(b); i++) {
			this->cs[d][i][0] += lambda * f_ab[i][0];
			this->cs[d][i][1] += lambda * f_ab[i][1];
		}

		// c
		for(int i = 0; i < dim; i++) {
			int ind = (h->H[d][i] * 3) & MASK2(b);
			int angle = (h->Sigma[d][i] * 3) & (HASH_OMEGA_PERIOD - 1);
			double tt = u[i] * u[i] * u[i];
			this->cs[d][ind][0] += lambda * scale_c * Hashes::Omega[angle][0] * tt;
			this->cs[d][ind][1] += lambda * scale_c * Hashes::Omega[angle][1] * tt;
		}
	}


	delete[] s_u;
	delete[] s_uu;
	delete[] f_ab;

}

// T(i,j,k) = lambda * u(i) * u(j) * u(k)
void CountSketch::add_sparse_rank_one_tensor(double lambda, int nnz, int* inds, double* values, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper, bool with_symmetric_adjustment) {

	double scale_a = 1.0 / 6;
	double scale_b = 3.0 / 6;
	double scale_c = 2.0 / 6;
	fftw_complex *s_u = new fftw_complex[POWER2(b)];
	fftw_complex *s_uu = new fftw_complex[POWER2(b)];
	fftw_complex *f_ab = new fftw_complex[POWER2(b)];
	fftw_complex t;

	// a and b
	for(int d = 0; d < B; d++) {

		memset(s_u, 0, sizeof(fftw_complex) * POWER2(b));
		for(int p = 0; p < nnz; p++) {
			int i = inds[p];
			double value = values[p];
			int ind = h->H[d][i];
			int angle = h->Sigma[d][i];
			s_u[ind][0] += Hashes::Omega[angle][0] * value;
			s_u[ind][1] += Hashes::Omega[angle][1] * value;
		}

		memset(s_uu, 0, sizeof(fftw_complex) * POWER2(b));
		for(int p = 0; p < nnz; p++) {
			int i = inds[p];
			double value = SQR(values[p]);
			int ind = (h->H[d][i] << 1) & MASK2(b);
			int angle = (h->Sigma[d][i] << 1) & (HASH_OMEGA_PERIOD - 1);
			s_uu[ind][0] += Hashes::Omega[angle][0] * value;
			s_uu[ind][1] += Hashes::Omega[angle][1] * value;
		}

		fft_wrapper->fft(s_u, s_u);
		fft_wrapper->fft(s_uu, s_uu);

		// the B part
		for(int i = 0; i < POWER2(b); i++) {
			f_ab[i][0] = scale_b * (s_uu[i][0] * s_u[i][0] - s_uu[i][1] * s_u[i][1]);
			f_ab[i][1] = scale_b * (s_uu[i][0] * s_u[i][1] + s_uu[i][1] * s_u[i][0]);
		}

		// the A part
		for(int i = 0; i < POWER2(b); i++) {
			t[0] = s_u[i][0]; t[1] = s_u[i][1];
			complex_mult(s_u[i], t, s_u[i]);
			complex_mult(s_u[i], t, s_u[i]);
			f_ab[i][0] += scale_a * s_u[i][0];
			f_ab[i][1] += scale_a * s_u[i][1];
		}

		// adding f_ab back to current count sketch
		ifft_wrapper->fft(f_ab, f_ab);
		for(int i = 0; i < POWER2(b); i++) {
			this->cs[d][i][0] += lambda * f_ab[i][0];
			this->cs[d][i][1] += lambda * f_ab[i][1];
		}

		// c
		for(int p = 0; p < nnz; p++) {
			int i = inds[p];
			double tt = values[p] * values[p] * values[p];
			int ind = (h->H[d][i] * 3) & MASK2(b);
			int angle = (h->Sigma[d][i] * 3) & (HASH_OMEGA_PERIOD - 1);
			this->cs[d][ind][0] += lambda * scale_c * Hashes::Omega[angle][0] * tt;
			this->cs[d][ind][1] += lambda * scale_c * Hashes::Omega[angle][1] * tt;
		}
	}


	delete[] s_u;
	delete[] s_uu;
	delete[] f_ab;

}

// T(i,i,j) = u(i) * v(j)
void CountSketch::add_degasym_rank_one_tensor(double lambda, double* u, double* v, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, bool with_symmetric_adjustment) {

	fftw_complex* s_u = new fftw_complex[POWER2(b)];
	fftw_complex* s_v = new fftw_complex[POWER2(b)];
	fftw_complex* f_ab = new fftw_complex[POWER2(b)];
	
	for(int d = 0; d < B; d++) {
	
		memset(s_u, 0, sizeof(fftw_complex) * POWER2(b));
		memset(s_v, 0, sizeof(fftw_complex) * POWER2(b));
		for(int i = 0; i < dim; i++) {
			int indv = h->H[d][i];
			int indu = (indv << 1) & MASK2(b);
			int anglev = h->Sigma[d][i];
			int angleu = (anglev << 1) & (HASH_OMEGA_PERIOD - 1);
			s_u[indu][0] += Hashes::Omega[angleu][0] * u[i];
			s_u[indu][1] += Hashes::Omega[angleu][1] * u[i];
			s_v[indv][0] += Hashes::Omega[anglev][0] * v[i];
			s_v[indv][1] += Hashes::Omega[anglev][1] * v[i];
		}
		
		fft_wrapper->fft(s_u, s_u);
		fft_wrapper->fft(s_v, s_v);
		for(int i = 0; i < POWER2(b); i++)
			complex_mult(s_u[i], s_v[i], f_ab[i]);
		ifft_wrapper->fft(f_ab, f_ab);
		for(int i = 0; i < POWER2(b); i++) {
			this->cs[d][i][0] += lambda * f_ab[i][0];
			this->cs[d][i][1] += lambda * f_ab[i][1];
		}
	
	}
	
	delete[] s_u;
	delete[] s_v;
	delete[] f_ab;

}

void CountSketch::add_sparse_degasym_rank_one_tensor(double lambda, int unnz, int* uinds, double* uvalues, int vnnz, int* vinds, double* vvalues, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, bool with_symmetric_adjustment) {

	fftw_complex* s_u = new fftw_complex[POWER2(b)];
	fftw_complex* s_v = new fftw_complex[POWER2(b)];
	fftw_complex* f_ab = new fftw_complex[POWER2(b)];
	
	for(int d = 0; d < B; d++) {
	
		memset(s_u, 0, sizeof(fftw_complex) * POWER2(b));
		memset(s_v, 0, sizeof(fftw_complex) * POWER2(b));
		
		for(int p = 0; p < unnz; p++) {
			int i = uinds[p];
			double value = uvalues[p];
			int ind = (h->H[d][i] << 1) & MASK2(b);
			int angle = (h->Sigma[d][i] << 1) & (HASH_OMEGA_PERIOD - 1);
			s_u[ind][0] += Hashes::Omega[angle][0] * value;
			s_u[ind][1] += Hashes::Omega[angle][1] * value;
		}
		
		for(int p = 0; p < vnnz; p++) {
			int i = vinds[p];
			double value = vvalues[p];
			int ind = h->H[d][i];
			int angle = h->Sigma[d][i];
			s_v[ind][0] += Hashes::Omega[angle][0] * value;
			s_v[ind][1] += Hashes::Omega[angle][1] * value;
		}
		
		fft_wrapper->fft(s_u, s_u);
		fft_wrapper->fft(s_v, s_v);
		for(int i = 0; i < POWER2(b); i++)
			complex_mult(s_u[i], s_v[i], f_ab[i]);
		ifft_wrapper->fft(f_ab, f_ab);
		for(int i = 0; i < POWER2(b); i++) {
			this->cs[d][i][0] += lambda * f_ab[i][0];
			this->cs[d][i][1] += lambda * f_ab[i][1];
		}
	
	}
	
	delete[] s_u;
	delete[] s_v;
	delete[] f_ab;

}

void CountSketch::add_semisparse_asym_rank_one_tensor(double lambda, int unnz, int* uinds, double* uvalues, double* v, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, bool with_symmetric_adjustment) {

    double scale_a = 1.0 / 6;
	double scale_b = 3.0 / 6;
	double scale_c = 2.0 / 6;
    
    fftw_complex* s_u = new fftw_complex[POWER2(b)];
    fftw_complex* s_v = new fftw_complex[POWER2(b)];
    fftw_complex* s_uu = new fftw_complex[POWER2(b)];
    fftw_complex* s_uv = new fftw_complex[POWER2(b)];
    fftw_complex* f_ab = new fftw_complex[POWER2(b)];
    fftw_complex t;
    
    for(int d = 0; d < B; d++) {
    
        // s_u
        memset(s_u, 0, sizeof(fftw_complex) * POWER2(b));
        for(int p = 0; p < unnz; p++) {
            int i = uinds[p];
            int ind = h->H[d][i];
            int angle = h->Sigma[d][i];
            s_u[ind][0] += Hashes::Omega[angle][0] * uvalues[p];
            s_u[ind][1] += Hashes::Omega[angle][1] * uvalues[p];
        }
        
        // s_v
        memset(s_v, 0, sizeof(fftw_complex) * POWER2(b));
        for(int i = 0; i < dim; i++) {
            int ind = h->H[d][i];
            int angle = h->Sigma[d][i];
            s_v[ind][0] += Hashes::Omega[angle][0] * v[i];
            s_v[ind][1] += Hashes::Omega[angle][1] * v[i];
        }
        
        // s_uu
        memset(s_uu, 0, sizeof(fftw_complex) * POWER2(b));
        for(int p = 0; p < unnz; p++) {
            int i = uinds[p];
            int ind = (h->H[d][i] << 1) & MASK2(b);
            int angle = (h->Sigma[d][i] << 1) & (HASH_OMEGA_PERIOD - 1);
            double value = SQR(uvalues[p]);
            s_uu[ind][0] += Hashes::Omega[angle][0] * value;
            s_uu[ind][1] += Hashes::Omega[angle][1] * value;
        }
        
        // s_uv
        memset(s_uv, 0, sizeof(fftw_complex) * POWER2(b));
        for(int p = 0; p < unnz; p++) {
            int i = uinds[p];
            int ind = (h->H[d][i] << 1) & MASK2(b);
            int angle = (h->Sigma[d][i] << 1) & (HASH_OMEGA_PERIOD - 1);
            double value = uvalues[p] * v[i];
            s_uv[ind][0] += Hashes::Omega[angle][0] * value;
            s_uv[ind][1] += Hashes::Omega[angle][1] * value;
        }
        
        fft_wrapper->fft(s_u, s_u);
        fft_wrapper->fft(s_v, s_v);
        fft_wrapper->fft(s_uu, s_uu);
        fft_wrapper->fft(s_uv, s_uv);
        
        memset(f_ab, 0, sizeof(fftw_complex) * POWER2(b));
        for(int i = 0; i < POWER2(b); i++) {
            complex_mult(s_u[i], s_u[i], t);
            complex_mult(s_v[i], t, t);
            f_ab[i][0] += 3 * scale_a * t[0];
            f_ab[i][1] += 3 * scale_a * t[1];
            complex_mult(s_uu[i], s_v[i], t);
            f_ab[i][0] += scale_b * t[0];
            f_ab[i][1] += scale_b * t[1];
            complex_mult(s_uv[i], s_u[i], t);
            f_ab[i][0] += 2 * scale_b * t[0];
            f_ab[i][1] += 2 * scale_b * t[1];
        }
        
        ifft_wrapper->fft(f_ab, f_ab);
        for(int i = 0; i < POWER2(b); i++) {
            this->cs[d][i][0] += lambda * f_ab[i][0];
            this->cs[d][i][1] += lambda * f_ab[i][1];
        }
        
        // c
        for(int p = 0; p < unnz; p++) {
            int i = uinds[p];
            int ind = (h->H[d][i] * 3) & MASK2(b);
            int angle = (h->Sigma[d][i] * 3) & (HASH_OMEGA_PERIOD - 1);
            double value = 3 * SQR(uvalues[p]) * v[i];
            this->cs[d][ind][0] += lambda * scale_c * Hashes::Omega[angle][0] * value;
            this->cs[d][ind][1] += lambda * scale_c * Hashes::Omega[angle][1] * value;
        }
    
    }
    
    delete[] s_u;
    delete[] s_v;
    delete[] s_uu;
    delete[] s_uv;
    delete[] f_ab;

}

void CountSketch::add_rank_one_tensor(double lambda, CountSketch* f_cs_u) {

    fftw_complex t;
    for(int d = 0; d < B; d++) {
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(f_cs_u->cs[d][i], t);
            complex_mult(t, f_cs_u->cs[d][i], t);
            complex_mult(t, f_cs_u->cs[d][i], t);
            cs[d][i][0] += lambda * t[0];
            cs[d][i][1] += lambda * t[1];
        }
    }

}

void CountSketch::add_asym_rank_one_tensor(double lambda, CountSketch* f_cs_u, CountSketch* f_cs_v) {

    fftw_complex t;
    for(int d = 0; d < B; d++) {
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(f_cs_u->cs[d][i], t);
            complex_mult(t, f_cs_u->cs[d][i], t);
            complex_mult(t, f_cs_v->cs[d][i], t);
            cs[d][i][0] += lambda * 3 * t[0];
            cs[d][i][1] += lambda * 3 * t[1];
        }
    }

}

void CountSketch::fft(FFT_wrapper *wrapper) {

	assert(wrapper->len == POWER2(b));
	for(int d = 0; d < B; d++) {
		wrapper->fft(cs[d], cs[d]);
	}

}

void CountSketch::init() {

	for(int d = 0; d < B; d++) {
		memset(cs[d], 0, sizeof(fftw_complex) * POWER2(b));
	}

}

AsymCountSketch::AsymCountSketch(Hashes* h) {
    
    this->B = h->B;
    this->b = h->b;
    this->order = 1;
    
    cs = new fftw_complex*[B];
    for(int d = 0; d < B; d++) {
        cs[d] = new fftw_complex[POWER2(b)];
        memset(cs[d], 0, sizeof(fftw_complex) * POWER2(b));
    }

    this->hs = new Hashes*[1];
    this->hs[0] = h;
    
    this->src_data = NULL;
    
}

AsymCountSketch::AsymCountSketch(int order, Hashes** hs) {

    this->B = hs[0]->B;
    this->b = hs[0]->b;
    this->order = order;
    for(int i = 0; i < order; i++)
        assert(this->B == hs[i]->B && this->b == hs[i]->b);
        
    cs = new fftw_complex*[B];
    for(int d = 0; d < B; d++) {
        cs[d] = new fftw_complex[POWER2(b)];
        memset(cs[d], 0, sizeof(fftw_complex) * POWER2(b));
    }
    
    this->hs = new Hashes*[order];
    for(int i = 0; i < order; i++) {
        this->hs[i] = hs[i];
    }
        
    this->src_data = NULL;

}

AsymCountSketch::~AsymCountSketch() {

    for(int d = 0; d < B; d++)
        delete[] cs[d];
    delete[] cs;
    delete[] hs;

}

void AsymCountSketch::set_tensor(Tensor* tensor) {

    init();
    int dim = tensor->dim;
    assert(order == 3);
    for(int i = 0; i < 3; i++)
        assert(dim == hs[i]->dim);
        
    for(int d = 0; d < B; d++) {
        for(int i = 0; i < dim; i++)
            for(int j = 0; j < dim; j++)
                for(int k = 0; k < dim; k++) {
                    int ind = (hs[0]->H[d][i] + hs[1]->H[d][j] + hs[2]->H[d][k]) & MASK2(b);
                    int omega = hs[0]->Sigma[d][i] * hs[1]->Sigma[d][j] * hs[2]->Sigma[d][k];
                    cs[d][ind][0] += omega * tensor->A[IND3D(i,j,k,dim)];
                }
    }
    
}

void AsymCountSketch::set_vector(double* u, int dim) {

    init();
    for(int i = 0; i < order; i++)
        assert(this->hs[i]->dim == dim);
    
    for(int d = 0; d < B; d++) {
        for(int i = 0; i < dim; i++) {
            int ind = 0, sigma = 1;
            for(int p = 0; p < order; p++) {
                ind += hs[p]->H[d][i];
                sigma *= hs[p]->Sigma[d][i];
            }
            ind = ind & MASK2(b);
            cs[d][ind][0] += sigma * u[i];
        }
    }

}

void AsymCountSketch::set_sparse_vector(int* inds, double* values, int nnz) {

    init();
    
    for(int d = 0; d < B; d++) {
        for(int j = 0; j < nnz; j++) {
            int i = inds[j];
            double value = values[j];
            int ind = 0, sigma = 1;
            for(int p = 0; p < order; p++) {
                ind += hs[p]->H[d][i];
                sigma *= hs[p]->Sigma[d][i];
            }
            ind = ind & MASK2(b);
            cs[d][ind][0] += sigma * value;            
        }
    }

}

void AsymCountSketch::add_entry(int* inds, double value) {

    for(int d = 0; d < B; d++) {
        int ind = 0, sigma = 1;
        for(int p = 0; p < order; p++) {
            ind += hs[p]->H[d][inds[p]];
            sigma *= hs[p]->Sigma[d][inds[p]];
        }
        ind = ind & MASK2(b);
        cs[d][ind][0] += sigma * value;
    }

}

void AsymCountSketch::add_rank_one_tensor(double lambda, int dim, AsymCountSketch* f_cs_u, AsymCountSketch* f_cs_v, AsymCountSketch* f_cs_w, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, bool do_ifft) {

    assert(f_cs_u->order == 1 && f_cs_v->order == 1 && f_cs_w->order == 1);
    assert(this->order == 3);
    for(int i = 0; i < 3; i++)
        assert(hs[i]->dim == dim);

    fftw_complex* tvec = new fftw_complex[POWER2(b)];
    
    for(int d = 0; d < B; d++) {
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(f_cs_u->cs[d][i], tvec[i]);
            complex_mult(tvec[i], f_cs_v->cs[d][i], tvec[i]);
            complex_mult(tvec[i], f_cs_w->cs[d][i], tvec[i]);
        }
        if (do_ifft) ifft_wrapper->fft(tvec, tvec);
        for(int i = 0; i < POWER2(b); i++) {
            cs[d][i][0] += lambda * tvec[i][0];
            cs[d][i][1] += lambda * tvec[i][1];
        }
    }
    
    delete[] tvec;

}

void AsymCountSketch::add_rank_one_tensor(double lambda, int dim, AsymCountSketch* f_cs_u, AsymCountSketch* f_cs_v, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, bool do_ifft) {

    assert(f_cs_u->order + f_cs_v->order == 3);
    assert(this->order == 3);
    for(int i = 0; i < 3; i++)
        assert(hs[i]->dim == dim);
        
    fftw_complex* tvec = new fftw_complex[POWER2(b)];
    
    for(int d = 0; d < B; d++) {
        for(int i = 0; i < POWER2(b); i++)
            complex_mult(f_cs_u->cs[d][i], f_cs_v->cs[d][i], tvec[i]);
        if (do_ifft) ifft_wrapper->fft(tvec, tvec);
        for(int i = 0; i < POWER2(b); i++) {
            cs[d][i][0] += lambda * tvec[i][0];
            cs[d][i][1] += lambda * tvec[i][1];
        }
    }
    
    delete[] tvec;

}

void AsymCountSketch::fft(FFT_wrapper* fft_wrapper) {

    assert(fft_wrapper->len == POWER2(b));
    for(int d = 0; d < B; d++)
        fft_wrapper->fft(cs[d], cs[d]);

}

void AsymCountSketch::init() {

    for(int d = 0; d < B; d++)
        memset(cs[d], 0, sizeof(fftw_complex) * POWER2(b));

}
