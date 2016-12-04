#ifndef COUNT_SKETCH_H_
#define COUNT_SKETCH_H_

#include "fftw3.h"
#include "hash.h"
#include "tensor.h"
#include "fft_wrapper.h"

class CountSketch {

public:
	int B, b;
	fftw_complex** cs;
	Hashes *h;
	void* src_data;

	CountSketch(Hashes*);
	~CountSketch();
	void copy_from(CountSketch*);
	
	double read_entry(int order, int* inds); // ind: int[order]

	void set_tensor(Tensor*, bool with_symmetric_adjustment = true);
	void set_vector(double*, int, int order = 1, bool with_symmetric_adjustment = true); // vec + len

    void add_entry(int order, int* inds, double value);
	
	void add_rank_one_tensor(double, double*, int, FFT_wrapper*, FFT_wrapper*, bool with_symmetric_adjustment = true); // lambda, u, dim, fft_wrapper, ifft_wrapper
	void add_sparse_rank_one_tensor(double, int, int*, double*, FFT_wrapper*, FFT_wrapper*, bool with_symmetric_adjustment = true); //lambda, nnz, inds, values, fft_wrapper, ifft_wrapper
	// T(i,j,k) = lambda * u(i) * u(j) * u(k)
	
	void add_degasym_rank_one_tensor(double, double*, double*, int, FFT_wrapper*, FFT_wrapper*, bool with_symmetric_adjustment = true); // lambda, u, v, dim, fft_wrapper, ifft_wrapper
	void add_sparse_degasym_rank_one_tensor(double, int, int*, double*, int, int*, double*, FFT_wrapper*, FFT_wrapper*, bool with_symmetric_adjustment = true); //lambda, sparse u, sparse v, ...
	// T(i, i, k) = lambda * u(i) * v(k)
    
    void add_asym_rank_one_tensor(double, double*, double*, int, FFT_wrapper*, FFT_wrapper*, bool with_symmetric_adjustment = true); // lambda, u, v, dim, fft_wrapper, ifft_wrapper
	void add_semisparse_asym_rank_one_tensor(double, int, int*, double*, double*, int, FFT_wrapper*, FFT_wrapper*, bool with_symmetric_adjustment = true); //lambda, sparse u, v, dim, ...
    // T(i, j, k) = lambda * (u(i) * u(j) * v(k) + u(i) * v(j) * u(k) + v(i) * u(j) * u(k))
    
    void add_rank_one_tensor(double, CountSketch*);
    void add_asym_rank_one_tensor(double, CountSketch*, CountSketch*);

	void fft(FFT_wrapper*);

private:
	void init();

};

class AsymCountSketch {

public:
    int B, b;
    int order;
    fftw_complex** cs;
    Hashes** hs;
    void* src_data;
    
    AsymCountSketch(Hashes* hs);
    AsymCountSketch(int order, Hashes** hs);
    ~AsymCountSketch();
    
    void set_tensor(Tensor*);
    // set_vector: cs[h_1(i)+h_2(i)+...] = s_1(i)s_2(i).. \cdot u[i]
    void set_vector(double*, int);
    void set_sparse_vector(int*, double*, int); // indices, values, nnz
    
    void add_entry(int*, double); // indices, value
    
    // scale, dim, f_cs_u, f_cs_v, f_cs_w, fft_wrapper, ifft_wrapper
    void add_rank_one_tensor(double, int, AsymCountSketch*, AsymCountSketch*, AsymCountSketch*, FFT_wrapper*, FFT_wrapper*, bool do_ifft = true);
    void add_rank_one_tensor(double, int, AsymCountSketch*, AsymCountSketch*, FFT_wrapper*, FFT_wrapper*, bool do_ifft = true);
    
    void fft(FFT_wrapper*);
    
private: 
    void init();

};

#endif
