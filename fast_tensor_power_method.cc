#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "config.h"
#include "util.h"
#include "hash.h"
#include "count_sketch.h"
#include "fft_wrapper.h"
#include "matlab_wrapper.h"

double fast_Tuuu(CountSketch* cs_T, CountSketch *f_cs_u, int dim, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper) {

	int B = cs_T->h->B;
	int b = cs_T->h->b;
	
	double* values = new double[B];
	fftw_complex* tvec = new fftw_complex[POWER2(b)];
	fftw_complex t, sum;

	for(int d = 0; d < B; d++) {
	
		for(int i = 0; i < POWER2(b); i++) {
			tvec[i][0] = t[0] = f_cs_u->cs[d][i][0]; 
			tvec[i][1] = t[1] = f_cs_u->cs[d][i][1];
			complex_mult(tvec[i], t, tvec[i]);
			complex_mult(tvec[i], t, tvec[i]);
		}
		ifft_wrapper->fft(tvec, tvec);
		sum[0] = sum[1] = 0;
		for(int i = 0; i < POWER2(b); i++) {
			t[0] = cs_T->cs[d][i][0];
			t[1] = cs_T->cs[d][i][1];
			sum[0] += tvec[i][0] * t[0] + tvec[i][1] * t[1];
			sum[1] += tvec[i][1] * t[0] - tvec[i][0] * t[1];
		}
		values[d] = sum[0];
	}


	qsort(values, B, sizeof(double), compare_double);

	double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);


	delete[] tvec;
	delete[] values;


	return ret;

}

void fast_TIuu(CountSketch* f_cs_T, CountSketch* f_cs_u, int dim, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper, double* ret) {

	int B = f_cs_T->h->B;
	int b = f_cs_T->h->b;
	clock_t start, stop;
	double* values = new double[dim*B];
	fftw_complex* tvec = new fftw_complex[POWER2(b)];
	fftw_complex t;
	double count_time=0;
	double vec_time=0;
	double fft_time=0;
	double for_time=0;
	double test=0;
	//printf("power2(b) = %d\n",POWER2(b));
	for(int d = 0; d < B; d++) {
		//start = clock();
		//for(int ii=0; ii<POWER2(b); ii++)
		//	for(int jj =0; jj<1000; jj++)
		//		for(int kk=0; kk<1000; kk++)
		//			test = test + 1;
		//stop = clock();
		//count_time = count_time + stop - start;
		
		// compute tvec
		//start = clock();
		for(int i = 0; i < POWER2(b); i++) {
			complex_assign(f_cs_T->cs[d][i], tvec[i]);
			complex_mult_conj(tvec[i], f_cs_u->cs[d][i], tvec[i]);
			complex_mult_conj(tvec[i], f_cs_u->cs[d][i], tvec[i]);
		}
		//stop = clock();		//add by zhao  
		//vec_time = vec_time +  stop-start;
		// inverse fft
		//start = clock();
		ifft_wrapper->fft(tvec, tvec);
		//stop = clock();		//add by zhao  
		//fft_time = fft_time +  stop-start;
    		

    		//start = clock();
		// reading off the elements
		for(int i = 0; i < dim; i++) {
			int ind = f_cs_T->h->H[d][i];
			int angle = f_cs_T->h->Sigma[d][i];
			values[IND2D(i, d, B)] = tvec[ind][0] * Hashes::Omega[angle][0] + tvec[ind][1] * Hashes::Omega[angle][1];
		}
		//stop = clock();		//add by zhao  
	}
	for(int i = 0; i < dim; i++) {
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		ret[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}

	delete[] values;
	delete[] tvec;

}

double fast_collide_Tuuu(CountSketch* cs_T, CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, double* u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper) {

    int B = cs_T->h->B;
    int b = cs_T->h->b;

    cs_u->fft(fft_wrapper);
    cs_uu->fft(fft_wrapper);
    double* values = new double[B];
    fftw_complex t;
    
    const double scale1 = 1.0 / 6;
    const double scale2 = 3.0 / 6;
    const double scale3 = 2.0 / 6;
    
    for(int d = 0; d < B; d++) {
        values[d] = 0;
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(f_cs_T->cs[d][i], t);
            complex_mult_conj(t, cs_u->cs[d][i], t);
            complex_mult_conj(t, cs_u->cs[d][i], t);
            complex_mult_conj(t, cs_u->cs[d][i], t);
            values[d] += scale1 * t[0];
            complex_assign(f_cs_T->cs[d][i], t);
            complex_mult_conj(t, cs_uu->cs[d][i], t);
            complex_mult_conj(t, cs_u->cs[d][i], t);
            values[d] += scale2 * t[0];                   
        }
        values[d] /= POWER2(b);
        for(int i = 0; i < dim; i++) {
            int ind = (3 * cs_T->h->H[d][i]) & MASK2(b);
            int offset = (3 * cs_T->h->Sigma[d][i]) & (HASH_OMEGA_PERIOD - 1);
            assert(0 <= offset && offset < 4);
            complex_mult_conj(cs_T->cs[d][ind], Hashes::Omega[offset], t);
            values[d] += scale3 * u[i] * u[i] * u[i] * t[0];
        }
    }
    
    double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);
    delete[] values;
    
    return ret;

}

void fast_collide_TIuu(CountSketch* cs_T, CountSketch* f_cs_T, CountSketch* cs_u, CountSketch* cs_uu, double* u, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret) {

    int B = cs_T->h->B;
    int b = cs_T->h->b;
    
    cs_u->fft(fft_wrapper);
    cs_uu->fft(fft_wrapper);
	double* values = new double[dim*B];
	memset(values, 0, sizeof(double) * dim*B);
	fftw_complex* tvec = new fftw_complex[POWER2(b)];
	fftw_complex t;    
	
	const double scale1 = 1.0 / 6;
	const double scale2 = 1.0 / 3;
	const double scale3 = 1.0 / 3;

    for(int d = 0; d < B; d++) {
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(cs_u->cs[d][i], t);
            complex_mult(t, cs_u->cs[d][i], t);
            complex_add(t, cs_uu->cs[d][i], t);
            complex_mult_conj(f_cs_T->cs[d][i], t, tvec[i]);
        }
        ifft_wrapper->fft(tvec, tvec);
        for(int i = 0; i < dim; i++) {
            int ind = cs_T->h->H[d][i];
            int omega = cs_T->h->Sigma[d][i];
            complex_mult_conj(tvec[ind], Hashes::Omega[omega], t);
            values[IND2D(i,d,B)] += scale1 * t[0];
        }
        for(int i = 0; i < POWER2(b); i++) {
            complex_mult_conj(f_cs_T->cs[d][i], cs_u->cs[d][i], tvec[i]);
        }
        ifft_wrapper->fft(tvec, tvec);
        for(int i = 0; i < dim; i++) {
            int ind = (cs_T->h->H[d][i] << 1) & MASK2(b);
            int omega = (cs_T->h->Sigma[d][i] << 1) & (HASH_OMEGA_PERIOD - 1);
            complex_mult_conj(tvec[ind], Hashes::Omega[omega], t);
            values[IND2D(i,d,B)] += scale2 * u[i] * t[0];
        }
        for(int i = 0; i < dim; i++) {
            int ind = (cs_T->h->H[d][i] * 3) & MASK2(b);
            int omega = (cs_T->h->Sigma[d][i] * 3) & (HASH_OMEGA_PERIOD - 1);
            complex_mult_conj(cs_T->cs[d][ind], Hashes::Omega[omega], t);
            values[IND2D(i,d,B)] += scale3 * SQR(u[i]) * t[0];
        }
    }
    
	for(int i = 0; i < dim; i++) {
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		ret[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}

	delete[] values;
	delete[] tvec;

}

void fast_TIuv(AsymCountSketch* f_cs_T, AsymCountSketch* f_cs_u2, AsymCountSketch* f_cs_u3, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* ret) {

    int B = f_cs_T->hs[0]->B;
    int b = f_cs_T->hs[0]->b;
    
	double* values = new double[dim*B];
	fftw_complex* tvec = new fftw_complex[POWER2(b)];
	fftw_complex t;

	for(int d = 0; d < B; d++) {
		// compute tvec
		for(int i = 0; i < POWER2(b); i++) {
			complex_assign(f_cs_T->cs[d][i], tvec[i]);
			complex_mult_conj(tvec[i], f_cs_u2->cs[d][i], tvec[i]);
			complex_mult_conj(tvec[i], f_cs_u3->cs[d][i], tvec[i]);
		}
		// inverse fft
		ifft_wrapper->fft(tvec, tvec);
		// reading off the elements
		for(int i = 0; i < dim; i++) {
			int ind = f_cs_T->hs[0]->H[d][i];
			int sigma = f_cs_T->hs[0]->Sigma[d][i];
			values[IND2D(i, d, B)] = sigma * tvec[ind][0];
		}
	}

	for(int i = 0; i < dim; i++) {
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		ret[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}

	delete[] values;
	delete[] tvec;

}

double fast_Tuvw(AsymCountSketch* f_cs_T, AsymCountSketch* f_cs_u1, AsymCountSketch* f_cs_u2, AsymCountSketch* f_cs_u3, int dim, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper) {

    int B = f_cs_T->hs[0]->B;
    int b = f_cs_T->hs[0]->b;
    
    double* values = new double[B];
    fftw_complex t, t_sum;
    
    for(int d = 0; d < B; d++) {
        t_sum[0] = t_sum[1] = 0;
        for(int i = 0; i < POWER2(b); i++) {
            complex_assign(f_cs_T->cs[d][i], t);
            complex_mult_conj(t, f_cs_u1->cs[d][i], t);
            complex_mult_conj(t, f_cs_u2->cs[d][i], t);
            complex_mult_conj(t, f_cs_u3->cs[d][i], t);
            complex_add(t_sum, t, t_sum);
        }
        values[d] = t_sum[0] / POWER2(b);
    }
    
    double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);
    delete[] values;
    
    return ret;

}

double fast_sqr_fnorm(AsymCountSketch* cs_T) {

    assert(cs_T->order == 3);
    int B = cs_T->B;
    int b = cs_T->b;
    
    double* values = new double[B];
    for(int d = 0; d < B; d++) {
        values[d] = 0;
        for(int i = 0; i < POWER2(b); i++)
            values[d] += SQR(cs_T->cs[d][i][0]);
    }
    
    qsort(values, B, sizeof(double), compare_double);
    double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)+1]);
    
    delete[] values;
    return ret;

}

double fast_sqr_fnorm(CountSketch* cs_T) {

    int B = cs_T->B;
    int b = cs_T->b;
    
    double* values = new double[B];
    for(int d = 0; d < B; d++) {
        values[d] = 0;
        for(int i = 0; i < POWER2(b); i++) 
            values[d] += complex_sqrnorm(cs_T->cs[d][i]);
    }
    
    qsort(values, B, sizeof(double), compare_double);
    double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)+1]);
    
    delete[] values;
    return ret;

}

//new method created by zhao
double random01()
{
	return ( rand() * 1.0 ) / RAND_MAX;
}
double random01_open()
{
	return ( rand() * 1.0 + 1.0 ) / (RAND_MAX + 1.0);
}
//new method created by zhao
void generate_list_of_sorted_random_numbers(double *a, int m)
{
	double sum=0;
	for(int i=0; i<m; i++)
	{
		sum = sum - log( random01_open() );
		a[i] = sum;
	}
	sum = sum - log( random01_open() );
	for(int i=0; i<m; i++)
		a[i] = a[i] / sum;
	//for(int i=0; i<m; i++)
	//	printf("%lf\n",a[i]);
	return;
}
//new method create by zhao
void assign_random_numbers_to_buckets(double *a, int m, double *q, int n, int *bucket)
{
	int i=0,j=1;
	while ( i < m )
	{
		if( a[i] <= q[j] )
		{
			bucket[i] = j - 1 ; i = i + 1;
		}
		else if(j < n )
		{
			j= j +1;
		}
		else
		{
			printf("index j= %d is too big, n = %d, i = %d, m = %d, a[i] = %f, q[j] = %f\n",j,n,i,m,a[i],q[j]);
			exit(-1);
		}
	}
	return;
}
//new method created by zhao
void generate_random_permutation(int *c, int m)
{
	for(int i=0; i<m; i++)
		c[i] = i;
	for(int i=0; i<m; i++)
	{
		double tmp = ( rand() * 1.0 ) / RAND_MAX;
		int t = (int) ( tmp * i );
		c[i] = c[t] ;
		c[t] = i;
	}
}
//new method created by zhao
void generate_cumulative_probablity(double *u, double *q, int n)
{
	q[0]=0;
	for(int i=1; i<=n; i++)
	{
		q[i] = q[i-1] + SQR(u[i-1]);
	}
	if(fabs(1-q[n]) > 1e-10)
		printf("inaccurary for q[n]\n");
	q[n]=1;
	return;
}


int binary_search(double target, double *q, int n)
{
	int low, mid, high;
	low = 0; 
	high = n;
	//mid = n / 2;
	while(high - low > 1)
	{
		mid = ( high + low )/2;
		if(target <= q[mid] )
			high = mid;
		else
			low = mid;
		
	}
	
	//printf("low = %d, mid = %d, high = %d, q[high] = %lf", low,mid,high,q[high]);
	if(mid==low)
		return mid;
	else
		return mid-1;
	//return mid-1;
}

double fast_sampling_binary_search_Tuuu(Tensor* T, double *u, int n, int B, int b)
{
	double *q = new double[n+1];
	size_t index_i, index_j, index_k;
	generate_cumulative_probablity(u,q,n);
	double* values = new double[B];
	for(int d=0; d<B; d++)
	{
		double sum=0;
		for(int i=0; i<b; i++)
		{
			index_i = binary_search (random01(),q,n);
			index_j = binary_search (random01(),q,n);
			index_k = binary_search (random01(),q,n);
			sum = sum + T->A[ IND3D(index_i,index_j,index_k,n) ] / ( u[index_i] * u[index_j] * u[index_k] );
		}
		values[d] = sum;
	}
	qsort(values, B, sizeof(double), compare_double);

	double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);
	ret = ret / (1.0 * b);
	delete [] values;
	delete [] q;
	return ret;
}

void fast_sampling_binary_search_TIuu(Tensor* T, double *u, int n, double* u_out, int B, int b)
{
	double *q = new double[n+1];
	size_t index_i, index_j, index_k;
	generate_cumulative_probablity(u,q,n);
	double* values = new double[n*B];
	for(int d=0; d<B; d++)
	{
		for(int j=0; j<n; j++)
		{
			double sum=0;
			for(int i=0; i<b; i++)
			{
				index_j = binary_search (random01(),q,n);
				index_k = binary_search (random01(),q,n);
				sum = sum + T->A[ IND3D(j,index_j,index_k,n) ] / (u[index_j] * u[index_k] );
			}
			values[IND2D(j, d, B)] = sum  / (1.0 * b);
		}	
	}
	for(int i = 0; i < n; i++) 
	{
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		u_out[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}
	
	delete [] values;
	delete [] q;
}

//new method created by zhao
double fast_sampling_linear_merge_Tuuu(Tensor* T, double *u, int n, int B, int b, int rank, double *old_lambda, double **old_v)
{
	double *q = new double[n+1];
	size_t index_i, index_j, index_k;
	generate_cumulative_probablity(u,q,n);
	b = b * n / 2;
	double **a;
	int **e;
	int **c;
	a = new double*[3];
	e = new int*[3];
	c = new int*[3];
	for(int i=0; i<3; i++)
	{
		a[i] = new double[b];
		e[i] = new int[b];
		c[i] = new int[b];
	}
	double* values = new double[B];
	for(int d=0; d<B; d++)
	{
		for(int i=0; i<3; i++)
		{
			generate_list_of_sorted_random_numbers(a[i],b);
			assign_random_numbers_to_buckets(a[i], b, q, n, e[i]);
			generate_random_permutation(c[i], b);
		}
		double sum=0;
		for(int i=0; i<b; i++)
		{
			index_i = e[0][ c[0][i] ];
			index_j = e[1][ c[1][i] ];
			index_k = e[2][ c[2][i] ];
			double theEntry = T->A[ IND3D(index_i,index_j,index_k,n) ]; //T->A[index];
			for(int k =0; k<=rank; k++)
				theEntry = theEntry - old_lambda[k] * old_v[k][index_i] * old_v[k][index_j] * old_v[k][index_k];
			sum = sum +  theEntry / ( u[index_i] * u[index_j] * u[index_k] );
		}
		values[d] = sum;
	}


	qsort(values, B, sizeof(double), compare_double);

	double ret = (B&1)? values[B>>1] : 0.5 * (values[B>>1] + values[(B>>1)-1]);

	//printf("vector : ");
	//for(int i=0; i<B; i++)
	//	printf("%lf,",values[i]);
	//printf("\n");
	//printf("ret = %lf\n",ret);
	ret = ret / (1.0 * b);
	
	for(int i=0; i<3; i++)
	{
		delete [] a[i];
		delete [] e[i];
		delete [] c[i];
	}
	delete [] a;
	delete [] e;
	delete [] c;
	delete [] q;
	delete [] values;
	return ret;
}

void generate_list_of_random_numbers(double *a, int m)
{
	for(int i=0; i<m; i++)
		a[i] = random01();
	return;
}

//new method created by zhao
void fast_sampling_linear_merge_TIuu(Tensor* T,double* u,int n,double* u_out, int B, int b, int rank, double *old_lambda, double **old_v)
{
	//printf("n = %d, B = %d, b = %d\n",n,B,b);
	double *q = new double[n+1];
	double *u_inverse = new double[n];
	for(int i =0; i<n; i++)
		u_inverse[i] = 1.0 / u[i];
	size_t index_j, index_k;
	generate_cumulative_probablity(u,q,n);
	//printf("output q\n");
	double **a;
	int **e;
	int **c;
	a = new double*[2];
	e = new int*[2];
	c = new int*[2];
	for(int i=0; i<2; i++)
	{
		a[i] = new double[b];
		e[i] = new int[b];
		c[i] = new int[b];
		
	}
	double* values = new double[n*B];
	for(int d=0; d<B; d++)
	{	
		for(int i=0; i<2; i++)
		{
			generate_list_of_sorted_random_numbers(a[i],b);
			assign_random_numbers_to_buckets(a[i], b, q, n, e[i]);
			generate_random_permutation(c[i], b);
		}
		
		double tmp;
		size_t index;
		size_t n2= n*n;
		size_t j3 = -n2;
		double sum;
		for(int j=0; j<n; j++)
		{
			sum = 0;
			j3 = j3+n2;
			for(int i=0; i<b; i++)
			{
				//#define IND3D(i,j,k,n) ((i)*(n)*(n)+(j)*(n)+(k))
				index_j = e[0][ c[0][i] ];
				index_k = e[1][ c[1][i] ];
				index = j3 + index_j*n + index_k ;
				double theEntry = T->A[index];
				for(int k =0; k<=rank; k++)
					theEntry = theEntry - old_lambda[k] * old_v[k][j] * old_v[k][index_j] * old_v[k][index_k];
				tmp = theEntry * u_inverse[index_j ] * u_inverse[index_k];
				sum = sum + tmp;
			} 
			values[IND2D(j, d, B)] = sum  / (1.0 * b);
		}
	}
	
	for(int i = 0; i < n; i++) 
	{
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		u_out[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}
	for(int i=0; i<2; i++)
	{
		delete [] a[i];
		delete [] e[i];
		delete [] c[i];
		
	}
	delete [] a;
	delete [] e;
	delete [] c;
	delete [] q;
	delete [] values;
	delete [] u_inverse;
	return;
}
//new method created by zhao
//input: u, output: u_out
//rank = -1 if true rank is 1
void fast_sampling_linear_merge_TIuu_dbg(Tensor* T,double* u,int n,double* u_out, int B, int b, int rank, double *old_lambda, double **old_v)
{
	static int entry = 0;
	// printf("rank = %d, n = %d, B = %d, b = %d\n",rank,n,B,b);
	double *q = new double[n+1];
	double *u_inverse = new double[n];
	for(int i =0; i<n; i++) {
		u_inverse[i] = 1.0 / u[i];
		// printf("u[%d]=%1.2g\t", i, u[i]);
        }
	// printf("-----------------------------------\n");
	size_t index_j, index_k;
	generate_cumulative_probablity(u,q,n);
	//printf("output q\n");
	double **a;
	int **e;
	int **c;
	a = new double*[2];
	e = new int*[2];
	c = new int*[2];
	for(int i=0; i<2; i++)
	{
		a[i] = new double[b];
		e[i] = new int[b];
		c[i] = new int[b];
		
	}
	double* values = new double[n*B];
	for(int d=0; d<B; d++)
	{	
		for(int i=0; i<2; i++)
		{
			generate_list_of_sorted_random_numbers(a[i],b);
			assign_random_numbers_to_buckets(a[i], b, q, n, e[i]);
			generate_random_permutation(c[i], b);
		}
		
		double tmp;
		size_t index;
		size_t n2= n*n;
		size_t j3 = -n2;
		double sum;
		for(int j=0; j<n; j++)
		{
			sum = 0;
			j3 = j3+n2;
			for(int i=0; i<b; i++)
			{
				//#define IND3D(i,j,k,n) ((i)*(n)*(n)+(j)*(n)+(k))
				index_j = e[0][ c[0][i] ];
				index_k = e[1][ c[1][i] ];
				if (j==0)printf("trial %d selecting index %lu, %lu\n", d, index_j, index_k);
				index = j3 + index_j*n + index_k ;
				double theEntry = T->A[index];
				for(int k =0; k<=rank; k++)
					theEntry = theEntry - old_lambda[k] * old_v[k][j] * old_v[k][index_j] * old_v[k][index_k];
				tmp = theEntry * u_inverse[index_j ] * u_inverse[index_k];
				sum = sum + tmp;
			} 
			values[IND2D(j, d, B)] = sum  / (1.0 * b);
		}
	}
	
	for(int i = 0; i < n; i++) 
	{
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		u_out[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}
	double* real_u_out = new double[n];
	T->TIuu(u, real_u_out);
	if (entry == 0) {
		for (int i = 0; i < n; ++i) {
			u_out[i] = real_u_out[i];
		}
		entry = 1;
	}
	double error = 0.0;
	for(int i = 0; i < n; ++i)
	{
		printf("%d:\t%8.5e\t%8.5e\t%8.5e\n", i, u[i], real_u_out[i], u_out[i]);
		double t = (real_u_out[i] - u_out[i]);
		error += t*t;
	}
	error = sqrt(error / n);
	printf("error = %f\n", error);
	printf("------------------------------------------\n");
	delete [] real_u_out;
	for(int i=0; i<2; i++)
	{
		delete [] a[i];
		delete [] e[i];
		delete [] c[i];
		
	}
	delete [] a;
	delete [] e;
	delete [] c;
	delete [] q;
	delete [] values;
	delete [] u_inverse;
	return;
}

void prescan_sampling_linear_merge_TIuu(Tensor* T,double* u,int n,double* u_out, int B, int* slice_b, int max_b, int rank, double *old_lambda, double **old_v)
{
	//printf("n = %d, B = %d, b = %d\n",n,B,b);
	double *q = new double[n+1];
	double *u_inverse = new double[n];
	for(int i =0; i<n; i++)
		u_inverse[i] = 1.0 / u[i];
	size_t index_j, index_k;
	generate_cumulative_probablity(u,q,n);
	//printf("output q\n");
	double **a;
	int **e;
	int **c;
	/*
	int total_b = 0;
	for(int i=0; i < n; ++i) {
		total_b += slice_b[i];
	}
	*/
	a = new double*[2];
	e = new int*[2];
	c = new int*[2];
	for(int i=0; i<2; i++)
	{
		a[i] = new double[max_b];
		e[i] = new int[max_b];
		c[i] = new int[max_b];
		
	}
	double* values = new double[n*B];
	for(int d=0; d<B; d++)
	{	
		for(int i=0; i<2; i++)
		{
			generate_list_of_sorted_random_numbers(a[i],max_b);
			assign_random_numbers_to_buckets(a[i], max_b, q, n, e[i]);
			generate_random_permutation(c[i], max_b);
		}
		
		double tmp;
		size_t index;
		size_t n2= n*n;
		size_t j3 = -n2;
		double sum;
		// int acc_b = 0;
		for(int j=0; j<n; j++)
		{
			sum = 0;
			j3 = j3+n2;
			// for(int i=acc_b; i<acc_b+slice_b[j]; i++)
			for(int i=0; i<slice_b[j]; i++)
			{
				//#define IND3D(i,j,k,n) ((i)*(n)*(n)+(j)*(n)+(k))
				index_j = e[0][ c[0][i] ];
				index_k = e[1][ c[1][i] ];
				index = j3 + index_j*n + index_k ;
				double theEntry = T->A[index];
				for(int k =0; k<=rank; k++)
					theEntry = theEntry - old_lambda[k] * old_v[k][j] * old_v[k][index_j] * old_v[k][index_k];
				tmp = theEntry * u_inverse[index_j ] * u_inverse[index_k];
				sum = sum + tmp;
			} 
			// acc_b += slice_b[j];
			if (slice_b[j] > 0)
				values[IND2D(j, d, B)] = sum  / (1.0 * slice_b[j]);
			else
				values[IND2D(j, d, B)] = 0.0;
		}
	}
	
	for(int i = 0; i < n; i++) 
	{
		double* base = values + i * B;
		qsort(base, B, sizeof(double), compare_double);
		u_out[i] = (B&1)? base[B>>1] : 0.5 * (base[B>>1] + base[(B>>1)-1]);
	}
	for(int i=0; i<2; i++)
	{
		delete [] a[i];
		delete [] e[i];
		delete [] c[i];
		
	}
	delete [] a;
	delete [] e;
	delete [] c;
	delete [] q;
	delete [] values;
	delete [] u_inverse;
	return;
}

double fast_sampling_Tuuu(Tensor* T, double *u, int n, int B, int b, int rank, double *old_lambda, double **old_v)
{
	return  fast_sampling_linear_merge_Tuuu(T, u, n,  B, b, rank, old_lambda, old_v);
	//return  fast_sampling_binary_search_Tuuu(T, u, n, B, b);
}


void fast_sampling_TIuu(Tensor* T, double *u, int n, double *u_out, int B, int b, int rank, double *old_lambda, double **old_v)
{
	fast_sampling_linear_merge_TIuu(T, u, n, u_out, B, b, rank, old_lambda, old_v);
	//fast_sampling_binary_search_TIuu(T, u, n, u_out, B, b);
	return;
}

void prescan_sampling_TIuu(Tensor* T, double *u, int n, double *u_out, int B, int* slice_b, int max_b, int rank, double *old_lambda, double **old_v)
{
	prescan_sampling_linear_merge_TIuu(T, u, n, u_out, B, slice_b, max_b, rank, old_lambda, old_v);
	return;
}

//new method created by zhao
void fast_sampling_tensor_power_method(Tensor* T, int dim, int rank, int L, int iterT, int B, int b,double *lambda, double **v)
{
	double *u = new double[dim];
	double *u_out = new double[dim];
	puts("--- Start sampling tensor power method ---");

	
	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);


		memset(v[k], 0, sizeof(double) * dim);
		double max_value = -1e100;

		for(int tau = 0; tau < L; tau++) {

			putchar('.');
			fflush(stdout);

			// Draw u randomly from the unit sphere and create its FFT count sketch
			generate_uniform_sphere_point(dim, u);

			for(int t = 0; t < iterT; t++) {
				//printf("t=%d\n",t);
				if (0)
					T->TIuu(u, u_out);
				else
					fast_sampling_TIuu(T,u,dim,u_out,B, b,k-1,lambda,v);//input is u, and output is v
				double norm = 0;
				for(int i = 0; i < dim; i++)
					norm += SQR(u_out[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim; i++)
					u[i] = u_out[i] * scale;
				//#ifdef DEBUG
				//if(tau <= 2)
				//printf("tau=%d, t=%d, Tuuu = %lf\n",tau,t,fast_sampling_Tuuu(T, u, dim, B, b));
				//#endif
			}

			// compute T(uuu) and update v[k]
			double value = fast_sampling_Tuuu(T, u, dim, B, b, k-1, lambda, v);
			if (value > max_value) { max_value = value; memcpy(v[k], u, sizeof(double) * dim);}

		}


		puts("#");
		fflush(stdout);
		printf("max_value = %lf\n", max_value);
		memcpy(u, v[k], sizeof(double) * dim);
		// Do another round of power update
		for(int t = 0; t < iterT; t++) {
			if (0)

				T->TIuu(u, u_out);
			else
				fast_sampling_TIuu(T, u, dim, u_out, B, b, k-1, lambda, v);
			double norm = 0;
			for(int i = 0; i < dim; i++)
				norm += SQR(u_out[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < dim; i++)
				u[i] = u_out[i] * scale;
			#ifdef DEBUG
			printf("t=%d, Tuuu = %lf\n",t,fast_sampling_Tuuu(T, u, dim, B, b, k-1, lambda, v));
			#endif
		}
		memcpy(v[k], u, sizeof(double) * dim);

		// compute the eigenvalue
		
		double tmp1 = fast_sampling_Tuuu(T, u, dim, B, b, k-1, lambda, v);
		//double tmp2 = T->Tuuu(u); 
		printf("lambda[%d] = %lf\n", k, tmp1);
		//printf("exact lambda[%d] = %lf\n", k, tmp2 );
		lambda[k]  = tmp1;

		// compute the deflated tensor
		//cs_T->add_rank_one_tensor(-lambda[k], v[k], dim, fft_wrapper, ifft_wrapper);

	}

	puts("Completed.");

	delete [] u;
	delete [] u_out;
	//delete f_cs_u;
	//delete f_cs_T;
}

void prescan_sampling_tensor_power_method(Tensor* T, int dim, int rank, int L, int iterT, int B, int b, int* slice_b, int max_b, double *lambda, double **v)
{
	double *u = new double[dim];
	double *u_out = new double[dim];
	puts("--- Start sampling tensor power method ---");
	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);


		memset(v[k], 0, sizeof(double) * dim);
		double max_value = -1e100;

		for(int tau = 0; tau < L; tau++) {

			putchar('.');
			fflush(stdout);

			// Draw u randomly from the unit sphere and create its FFT count sketch
			generate_uniform_sphere_point(dim, u);

			for(int t = 0; t < iterT; t++) {
				//printf("t=%d\n",t);
				if (0)
					T->TIuu(u, u_out);
				else
					prescan_sampling_TIuu(T,u,dim,u_out,B, slice_b,max_b,k-1,lambda,v);//input is u, and output is v
				double norm = 0;
				for(int i = 0; i < dim; i++)
					norm += SQR(u_out[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim; i++)
					u[i] = u_out[i] * scale;
				//#ifdef DEBUG
				//if(tau <= 2)
				//printf("tau=%d, t=%d, Tuuu = %lf\n",tau,t,fast_sampling_Tuuu(T, u, dim, B, b));
				//#endif
			}

			// compute T(uuu) and update v[k]
			double value = fast_sampling_Tuuu(T, u, dim, B, b, k-1, lambda, v);
			if (value > max_value) { max_value = value; memcpy(v[k], u, sizeof(double) * dim);}

		}


		puts("#");
		fflush(stdout);
		printf("max_value = %lf\n", max_value);
		memcpy(u, v[k], sizeof(double) * dim);
		// Do another round of power update
		for(int t = 0; t < iterT; t++) {
			if (0)

				T->TIuu(u, u_out);
			else
				fast_sampling_TIuu(T, u, dim, u_out, B, b, k-1, lambda, v);
			double norm = 0;
			for(int i = 0; i < dim; i++)
				norm += SQR(u_out[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < dim; i++)
				u[i] = u_out[i] * scale;
			#ifdef DEBUG
			printf("t=%d, Tuuu = %lf\n",t,fast_sampling_Tuuu(T, u, dim, B, b, k-1, lambda, v));
			#endif
		}
		memcpy(v[k], u, sizeof(double) * dim);

		// compute the eigenvalue
		
		double tmp1 = fast_sampling_Tuuu(T, u, dim, B, b, k-1, lambda, v);
		//double tmp2 = T->Tuuu(u); 
		printf("lambda[%d] = %lf\n", k, tmp1);
		//printf("exact lambda[%d] = %lf\n", k, tmp2 );
		lambda[k]  = tmp1;

		// compute the deflated tensor
		//cs_T->add_rank_one_tensor(-lambda[k], v[k], dim, fft_wrapper, ifft_wrapper);

	}

	puts("Completed.");

	delete [] u;
	delete [] u_out;
	//delete f_cs_u;
	//delete f_cs_T;
}

void fast_tensor_power_method(CountSketch *cs_T, int dim, int rank, int L, int T, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper, double *lambda, double **v) {

	CountSketch *f_cs_T = new CountSketch(cs_T->h);
	CountSketch *f_cs_u = new CountSketch(cs_T->h);
	double *u = new double[dim];

	puts("--- Start fast tensor power method ---");

	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);

		// create FFT of cs_T
		f_cs_T->copy_from(cs_T);
		f_cs_T->fft(fft_wrapper);

		memset(v[k], 0, sizeof(double) * dim);
		double max_value = -1e100;

		for(int tau = 0; tau < L; tau++) {

			putchar('.');
			fflush(stdout);

			// Draw u randomly from the unit sphere and create its FFT count sketch
			generate_uniform_sphere_point(dim, u);
			f_cs_u->set_vector(u, dim);
			f_cs_u->fft(fft_wrapper);

			for(int t = 0; t < T; t++) {
				fast_TIuu(f_cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper, u);
				double norm = 0;
				for(int i = 0; i < dim; i++)
					norm += SQR(u[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim; i++)
					u[i] *= scale;
				f_cs_u->set_vector(u, dim);
				f_cs_u->fft(fft_wrapper);
				//#ifdef DEBUG
				//if(tau <=2)
				//	printf("tau=%d,t=%d,Tuuu=%lf\n",tau,t,fast_Tuuu(cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper));
				//#endif
			}

			// compute T(uuu) and update v[k]
			double value = fast_Tuuu(cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper);
			if (value > max_value) { max_value = value; memcpy(v[k], u, sizeof(double) * dim);}
			//printf("tau=%d, Tuuu=%e\n",tau,value);
		}

		puts("#");
		fflush(stdout);
		printf("max_value = %lf\n", max_value);
		// Do another round of power update
		f_cs_u->set_vector(v[k], dim);
		f_cs_u->fft(fft_wrapper);
		for(int t = 0; t < T; t++) {
			fast_TIuu(f_cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper, u);
			double norm = 0;
			for(int i = 0; i < dim; i++)
				norm += SQR(u[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < dim; i++)
				u[i] *= scale;
			f_cs_u->set_vector(u, dim);
			f_cs_u->fft(fft_wrapper);
			#if DEBUG
			printf("t=%d,Tuuu=%lf\n",t,fast_Tuuu(cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper));
			#endif
		}
		memcpy(v[k], u, sizeof(double) * dim);

		// compute the eigenvalue
		lambda[k] = fast_Tuuu(cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper);
		printf("lambda[%d] = %lf\n", k, lambda[k]);

		// compute the deflated tensor
		cs_T->add_rank_one_tensor(-lambda[k], v[k], dim, fft_wrapper, ifft_wrapper);

	}

	puts("Completed.");

	delete[] u;
	delete f_cs_u;
	delete f_cs_T;

}

double fast_collide_tensor_power_method(CountSketch* cs_T, int dim, int rank, int L, int T, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double** v, bool report_residue) {

	CountSketch *f_cs_T = new CountSketch(cs_T->h);
	CountSketch *cs_u = new CountSketch(cs_T->h);
	CountSketch *cs_uu = new CountSketch(cs_T->h);
	double *u = new double[dim];
	
	double t_fnorm = 1;
	double residue = 0;
	
	if (report_residue) {
	    t_fnorm = fast_sqr_fnorm(cs_T);
	    printf("Before tensor power method: fnorm = %lf\n", t_fnorm);
	}

	puts("--- Start fast collide tensor power method ---");

	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);

		// create FFT of cs_T
		f_cs_T->copy_from(cs_T);
		f_cs_T->fft(fft_wrapper);

		memset(v[k], 0, sizeof(double) * dim);
		double max_value = -1e100;

		for(int tau = 0; tau < L; tau++) {

			putchar('.');
			fflush(stdout);

			// Draw u randomly from the unit sphere and create its FFT count sketch
			generate_uniform_sphere_point(dim, u);
			cs_u->set_vector(u, dim, 1);
			cs_uu->set_vector(u, dim, 2);

			for(int t = 0; t < T; t++) {
				fast_collide_TIuu(cs_T, f_cs_T, cs_u, cs_uu, u, dim, fft_wrapper, ifft_wrapper, u);
				double norm = 0;
				for(int i = 0; i < dim; i++)
					norm += SQR(u[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim; i++)
					u[i] *= scale;
                cs_u->set_vector(u, dim, 1);
                cs_uu->set_vector(u, dim, 2);
			}

			// compute T(uuu) and update v[k]
			double value = fast_collide_Tuuu(cs_T, f_cs_T, cs_u, cs_uu, u, dim, fft_wrapper, ifft_wrapper);
			if (value > max_value) { max_value = value; memcpy(v[k], u, sizeof(double) * dim);}

		}

		puts("#");
		fflush(stdout);

		// compute the eigenvalue
		lambda[k] = max_value;

		// compute the deflated tensor
		cs_T->add_rank_one_tensor(-lambda[k], v[k], dim, fft_wrapper, ifft_wrapper, false);

	}

	puts("Completed.");
	
	if (report_residue) {
	    residue = fast_sqr_fnorm(cs_T);
	    printf("After tensor power method: fnorm = %lf\n", residue);
	}

	delete[] u;
	delete f_cs_T;
	delete cs_u;
	delete cs_uu;
	
	return residue / t_fnorm;

}

void fast_kernel_tensor_power_method(CountSketch *cs_T, int dim1, int dim2, double* W, int rank, int L, int T, FFT_wrapper *fft_wrapper, FFT_wrapper *ifft_wrapper, Matlab_wrapper* mat_wrapper, double *lambda, double **v) {

	CountSketch *f_cs_T = new CountSketch(cs_T->h);
	CountSketch *f_cs_u = new CountSketch(cs_T->h);
	double *u = new double[dim2];
	double *wu = new double[dim1];

	puts("--- Start fast kernel tensor power method ---");

	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);

		// create FFT of cs_T
		f_cs_T->copy_from(cs_T);
		f_cs_T->fft(fft_wrapper);

		memset(v[k], 0, sizeof(double) * dim2);
		double max_value = -1e100;

		for(int tau = 0; tau < L; tau++) {

			putchar('.');
			fflush(stdout);

			// Draw u randomly from the unit sphere and create its FFT count sketch
			generate_uniform_sphere_point(dim2, u);
			mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u
			f_cs_u->set_vector(wu, dim1);
			f_cs_u->fft(fft_wrapper);

			for(int t = 0; t < T; t++) {
				fast_TIuu(f_cs_T, f_cs_u, dim1, fft_wrapper, ifft_wrapper, wu);
				mat_wrapper->multiply(W, wu, dim1, dim2, dim1, 1, dim2, 1, true, false, u); // u = W' * wu
				double norm = 0;
				for(int i = 0; i < dim2; i++)
					norm += SQR(u[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim2; i++)
					u[i] *= scale;
				mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u
				f_cs_u->set_vector(wu, dim1);
				f_cs_u->fft(fft_wrapper);
			}

			// compute T(uuu) and update v[k]
			double value = fast_Tuuu(cs_T, f_cs_u, dim1, fft_wrapper, ifft_wrapper);
			if (value > max_value) { max_value = value; memcpy(v[k], u, sizeof(double) * dim2);}

		}

		puts("#");
		fflush(stdout);

		// Do another round of power update
		mat_wrapper->multiply(W, v[k], dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u
		f_cs_u->set_vector(wu, dim1);
		f_cs_u->fft(fft_wrapper);
		for(int t = 0; t < T; t++) {
			fast_TIuu(f_cs_T, f_cs_u, dim1, fft_wrapper, ifft_wrapper, wu);
			mat_wrapper->multiply(W, wu, dim1, dim2, dim1, 1, dim2, 1, true, false, u); // u = W' * wu
			double norm = 0;
			for(int i = 0; i < dim2; i++)
				norm += SQR(u[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < dim2; i++)
				u[i] *= scale;
			mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u
			f_cs_u->set_vector(wu, dim1);
			f_cs_u->fft(fft_wrapper);
		}
		memcpy(v[k], u, sizeof(double) * dim2);

		// compute the eigenvalue
		lambda[k] = fast_Tuuu(cs_T, f_cs_u, dim1, fft_wrapper, ifft_wrapper);
//		printf("lambda[%d] = %lf\n", k, lambda[k]);

		// compute the deflated tensor
		mat_wrapper->mldivide(W, v[k], dim1, dim2, true, wu); // wu = W' \ u
		cs_T->add_rank_one_tensor(-lambda[k], wu, dim1, fft_wrapper, ifft_wrapper);

	}

	puts("Completed.");

	delete[] u;
	delete[] wu;
	delete f_cs_u;
	delete f_cs_T;

}

void slow_tensor_power_method(Tensor* tensor, int dim, int rank, int L, int T, double* lambda, double** v) {

	double *u = new double[dim];
	double *w = new double[dim];

	puts("--- Start slow tensor power method ---");

	for(int k = 0; k < rank; k++) {

		printf("For rank %d: ", k);

		memset(v[k], 0, sizeof(double) * dim);
		double max_value = -1e100;
		for(int tau = 0; tau < L; tau++) {
			putchar('.');
			fflush(stdout);
			generate_uniform_sphere_point(dim, u);
			for(int t = 0; t < T; t++) {
				tensor->TIuu(u, w);
				double norm = 0;
				for(int i = 0; i < dim; i++) norm += SQR(w[i]);
				double scale = 1.0 / sqrt(norm);
				for(int i = 0; i < dim; i++) w[i] *= scale;
				memcpy(u, w, sizeof(double) * dim);
			}
			double value = tensor->Tuuu(u);
			if (value > max_value) {
				max_value = value;
				memcpy(v[k], u, sizeof(double) * dim);
			}
		}
		puts("#");
		fflush(stdout);
		printf("max_value = %lf\n", max_value);

		// Do another T round of update
		for(int t = 0; t < T; t++) {
			tensor->TIuu(v[k], w);
			double norm = 0;
			for(int i = 0; i < dim; i++) norm += SQR(w[i]);
			double scale = 1.0 / sqrt(norm);
			for(int i = 0; i < dim; i++) w[i] *= scale;
			memcpy(v[k], w, sizeof(double) * dim);
		}

		// Compute the eigenvalue
		lambda[k] = tensor->Tuuu(v[k]);
		printf("lambda[%d] = %lf\n", k, lambda[k]);

		// Compute the deflated tensor
		tensor->add_rank_one_update(-lambda[k], v[k]);
	}

	puts("Completed.");

	delete[] u;
	delete[] w;

}

void slow_kernel_tensor_power_method(Tensor* tensor, size_t dim1, size_t dim2, double* W, int rank, int L, int T, Matlab_wrapper* mat_wrapper, double* lambda, double** v) {

    double* u = new double[dim2];
    double* wu = new double[dim1];
    double* wv = new double[dim1];
    
    puts("-- Start slow kernel tensor power method --");
    
    for(int k = 0; k < rank; k++) {
    
        printf("For rank %d: ", k);
        
        memset(v[k], 0, sizeof(double) * dim2);
        double max_value = 1e-100;
        for(int tau = 0; tau < L; tau++) {
            putchar('.');
            fflush(stdout);
            generate_uniform_sphere_point(dim2, u);
            mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu); // wu = W * u      
            for(int t = 0; t < T; t++) {
                tensor->TIuu(wu, wv);
                mat_wrapper->multiply(W, wv, dim1, dim2, dim1, 1, dim2, 1, true, false, u); 
                double norm = 0;
                for(int i = 0; i < dim2; i++) norm += SQR(u[i]);
                double scale = 1.0 / sqrt(norm);
                for(int i = 0; i < dim2; i++) u[i] *= scale;
                mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu);
            }
            double value = tensor->Tuuu(wu);
            if (value > max_value) {
                max_value = value;
                memcpy(v[k], u, sizeof(double) * dim2);
            }
        }
        puts("");
        
        // do another T rounds
        mat_wrapper->multiply(W, v[k], dim1, dim2, dim2, 1, dim1, 1, false, false, wu);
        for(int t = 0; t < T; t++) {
            tensor->TIuu(wu, wv);
            mat_wrapper->multiply(W, wv, dim1, dim2, dim1, 1, dim2, 1, true, false, u); 
            double norm = 0;
            for(int i = 0; i < dim2; i++) norm += SQR(u[i]);
            double scale = 1.0 / sqrt(norm);
            for(int i = 0; i < dim2; i++) u[i] *= scale;
            mat_wrapper->multiply(W, u, dim1, dim2, dim2, 1, dim1, 1, false, false, wu);        
        }
        
        memcpy(v[k], u, sizeof(double) * dim2);
        lambda[k] = tensor->Tuuu(wu);
        
        mat_wrapper->mldivide(W, v[k], dim1, dim2, true, wu);
        for(size_t i1 = 0; i1 < dim1; i1++)
            for(size_t i2 = 0; i2 < dim1; i2++)
                for(size_t i3 = 0; i3 < dim1; i3++)
                    tensor->A[IND3D(i1,i2,i3,dim1)] -= lambda[k] * wu[i1] * wu[i2] * wu[i3];
        
    }
    
    delete[] u;
    delete[] wu;
    delete[] wv;

}

void slow_ALS(Tensor* tensor, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, double* lambda, double* A) {

    puts("Start slow ALS ...");

    double* XCB = new double[rank * dim];
    double* AA = new double[rank * rank];
    
    for(int t = 0; t < T; t++) {
        
        //printf("ALS Round %d ...\n", t);
        
        for(int k = 0; k < rank; k++) {
            tensor->TIuu(A + k * dim, XCB + k * dim);
        }
        
        mat_wrapper->multiply(A, A, rank, dim, rank, dim, rank, rank, false, true, AA);
        for(int i = 0; i < rank * rank; i++)
            AA[i] *= AA[i];
        mat_wrapper->pinv(AA, rank, rank, AA);
        mat_wrapper->multiply(AA, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
        
        // normalize each row of A
        for(int k = 0; k < rank; k++) {
            lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
            double scale = 1.0 / lambda[k];
            for(int i = 0; i < dim; i++)
                A[IND2D(k,i,dim)] *= scale;
        }
        
    }
    
    delete[] XCB;
    delete[] AA;
    
}

void slow_asym_ALS_update(Tensor* tensor, int dim, int rank, Matlab_wrapper* mat_wrapper, double* lambda, double* A, double* B, double* C, double* XCB, double* CC, double* BB) {

    for(int k = 0; k < rank; k++) {
        tensor->TIuv(B + k * dim, C + k * dim, XCB + k * dim);
    }   
    
    mat_wrapper->multiply(B, B, rank, dim, rank, dim, rank, rank, false, true, BB);
    mat_wrapper->multiply(C, C, rank, dim, rank, dim, rank, rank, false, true, CC);
    for(int p = 0; p < rank*rank; p++)
        BB[p] *= CC[p];
    mat_wrapper->pinv(BB, rank, rank, BB);
    mat_wrapper->multiply(BB, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
    
    // normalize each row of A
    for(int k = 0; k < rank; k++) {
        lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
        double scale = 1.0 / lambda[k];
        for(int i = 0; i < dim; i++)
            A[IND2D(k,i,dim)] *= scale;
    }

}

void slow_asym_ALS(Tensor* tensor, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, double* lambda, double* A, double* B, double* C) {

    printf("Slow Asym ALS: dim = %d, rank = %d\n", dim, rank);
    
    double* XCB = new double[rank * dim];
    double* CC = new double[rank * rank];
    double* BB = new double[rank * rank];

    for(int t = 0; t < T; t++) {
        
        printf("Slow asym round %d ...\n", t);
        
        slow_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, A, B, C, XCB, CC, BB);
        slow_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, B, C, A, XCB, CC, BB);
        slow_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, C, A, B, XCB, CC, BB);
        
    }
    
    delete[] XCB;
    delete[] CC;
    delete[] BB;

}

void fast_ALS(CountSketch* cs_T, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double* A) {

    puts("Start Fast ALS ...");
    
    double* XCB = new double[rank * dim];
    double* AA = new double[rank * rank];
    
    cs_T->fft(fft_wrapper);
    CountSketch *f_cs_u = new CountSketch(cs_T->h);
    
    for(int t = 0; t < T; t++) {
        
        printf("Fast ALS Round %d ...\n", t);
        
        for(int k = 0; k < rank; k++) {
            f_cs_u->set_vector(A + k * dim, dim);
            f_cs_u->fft(fft_wrapper);
            fast_TIuu(cs_T, f_cs_u, dim, fft_wrapper, ifft_wrapper, XCB + k * dim);
        }
        
        mat_wrapper->multiply(A, A, rank, dim, rank, dim, rank, rank, false, true, AA);
        for(int i = 0; i < rank * rank; i++)
            AA[i] *= AA[i];
        mat_wrapper->pinv(AA, rank, rank, AA);
        mat_wrapper->multiply(AA, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
        
        // normalize each row of A
        for(int k = 0; k < rank; k++) {
            lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
            double scale = 1.0 / lambda[k];
            for(int i = 0; i < dim; i++)
                A[IND2D(k,i,dim)] *= scale;
        }        
        
    }
    
    cs_T->fft(ifft_wrapper);
    delete[] XCB;
    delete[] AA;
    delete f_cs_u;

}

void fast_asym_ALS_update(AsymCountSketch* f_cs_T, int dim, int rank, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, 
    double* lambda, double* A, double* B, double* C, double* XCB, double* CC, double* BB, AsymCountSketch* cs_u2, AsymCountSketch* cs_u3) {
    
    for(int k = 0; k < rank; k++) {
        cs_u2->set_vector(B + k * dim, dim);
        cs_u3->set_vector(C + k * dim, dim);
        cs_u2->fft(fft_wrapper);
        cs_u3->fft(fft_wrapper);
        fast_TIuv(f_cs_T, cs_u2, cs_u3, dim, fft_wrapper, ifft_wrapper, XCB + k * dim);
    }
    
    /*for(int k = 0; k < rank; k++) {
        for(int i = 0; i < dim; i++)
            printf("%lf ", XCB[IND2D(k,i,dim)]);
        puts("");
    } */  
    
    mat_wrapper->multiply(B, B, rank, dim, rank, dim, rank, rank, false, true, BB);
    mat_wrapper->multiply(C, C, rank, dim, rank, dim, rank, rank, false, true, CC);
    for(int p = 0; p < rank*rank; p++)
        BB[p] *= CC[p];
    mat_wrapper->pinv(BB, rank, rank, BB);
    mat_wrapper->multiply(BB, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
    
    // normalize each row of A
    for(int k = 0; k < rank; k++) {
        lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
        double scale = 1.0 / lambda[k];
        for(int i = 0; i < dim; i++)
            A[IND2D(k,i,dim)] *= scale;
    }    
    
}

void fast_sample_asym_ALS_TIuv_update(Tensor* T, double* u, double *v, double* u_out, int n, int para_B, int para_b)
{
	//printf("n = %d, B = %d, b = %d\n",n,B,b);
	double **q = new double*[2];
	q[0] = new double[n+1];
	q[1] = new double[n+1];
	//double *p = new double[n+1];
	double *u_inverse = new double[n];
	double *v_inverse = new double[n];
	for(int i =0; i<n; i++)
	{
		u_inverse[i] = 1.0 / u[i];
		v_inverse[i] = 1.0 / v[i];
	}
	size_t index_j, index_k;
	generate_cumulative_probablity(u,q[0],n);
	generate_cumulative_probablity(v,q[1],n);
	//printf("output q\n");
	double **a;
	int **e;
	int **c;
	a = new double*[2];
	e = new int*[2];
	c = new int*[2];
	for(int i=0; i<2; i++)
	{
		a[i] = new double[para_b];
		e[i] = new int[para_b];
		c[i] = new int[para_b];
	}
	double* values = new double[n*para_B];
	for(int d=0; d<para_B; d++)
	{	
		for(int i=0; i<2; i++)
		{
			generate_list_of_sorted_random_numbers(a[i], para_b);
			assign_random_numbers_to_buckets(a[i], para_b, q[i], n, e[i]);
			generate_random_permutation(c[i], para_b);
		}
		
		double tmp;
		size_t index;
		double sum;
		double theEntry;
		for(int index_i = 0; index_i < n; index_i++)
		{
			sum = 0;
			for(int i=0; i<para_b; i++)
			{
				//#define IND3D(i,j,k,n) ((i)*(n)*(n)+(j)*(n)+(k))
				index_j = e[0][ c[0][i] ];
				index_k = e[1][ c[1][i] ];
				index = IND3D(index_i, index_j, index_k, n);
				theEntry = T->A[index];
				tmp = theEntry * u_inverse[ index_j ] * v_inverse[ index_k ];
				sum = sum + tmp;
			} 
			values[IND2D(index_i, d, para_B)] = sum  / (1.0 * para_b);
		}
	}
	
	for(int i = 0; i < n; i++) 
	{
		double* base = values + i * para_B;
		qsort(base, para_B, sizeof(double), compare_double);
		u_out[i] = (para_B&1)? base[para_B>>1] : 0.5 * (base[para_B>>1] + base[(para_B>>1)-1]);
	}
	for(int i=0; i<2; i++)
	{
		delete [] a[i];
		delete [] e[i];
		delete [] c[i];
		delete [] q[i];
	}
	delete [] a;
	delete [] e;
	delete [] c;
	delete [] q;
	delete [] values;
	delete [] u_inverse;
	delete [] v_inverse;
	return;
}

void normalize_vector(double *input_vector, double *output_vector, int dim)
{
	double norm = 0;
	for(int i = 0; i < dim; i++)
		norm += SQR(input_vector[i]);
	double scale = 1.0 / sqrt(norm);
	for(int i = 0; i < dim; i++)
		output_vector[i] = input_vector[i] * scale;
}

void fast_sample_asym_ALS_update(Tensor* tensor, int dim, int rank, Matlab_wrapper* mat_wrapper, double* lambda, double* A, double* B, double* C, double* XCB, double* CC, double* BB, int para_B, int para_b) {

    for(int k = 0; k < rank; k++) {
        // tensor->TIuv(B + k * dim, C + k * dim, XCB + k * dim);
	fast_sample_asym_ALS_TIuv_update(tensor, B + k * dim, C + k * dim, XCB + k * dim, dim, para_B, para_b);
    }   
    
    mat_wrapper->multiply(B, B, rank, dim, rank, dim, rank, rank, false, true, BB);
    mat_wrapper->multiply(C, C, rank, dim, rank, dim, rank, rank, false, true, CC);
    for(int p = 0; p < rank*rank; p++)
        BB[p] *= CC[p];
    mat_wrapper->pinv(BB, rank, rank, BB);
    mat_wrapper->multiply(BB, XCB, rank, rank, rank, dim, rank, dim, true, false, A);
    
    // normalize each row of A
    for(int k = 0; k < rank; k++) {
        lambda[k] = sqrt(vector_sqr_fnorm(A + k * dim, dim));
        double scale = 1.0 / lambda[k];
        for(int i = 0; i < dim; i++)
            A[IND2D(k,i,dim)] *= scale;
    }

}

double fast_sample_asym_ALS(Tensor* tensor, int dim, int rank, int para_T, int para_B, int para_b, Matlab_wrapper* mat_wrapper, double * lambda, double *A, double *B, double *C)
{
	
	puts("--- Start sampling ALS method ---");
	double* XCB = new double[rank * dim];
	double* CC = new double[rank * rank];
	double* BB = new double[rank * rank];

	for(int t = 0; t < para_T; t++) {
        	printf("sampling asym round %d ...\n", t);
		fast_sample_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, A, B, C, XCB, CC, BB, para_B, para_b);
		fast_sample_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, B, C, A, XCB, CC, BB, para_B, para_b);
		fast_sample_asym_ALS_update(tensor, dim, rank, mat_wrapper, lambda, C, A, B, XCB, CC, BB, para_B, para_b);
	}
	puts("Completed.");
}

double fast_asym_ALS(AsymCountSketch* cs_T, int dim, int rank, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double* A, double* B, double* C, bool report_residue) {

    double residue = 0;
    double t_fnorm = 0;
    
    if (report_residue) {
        t_fnorm = fast_sqr_fnorm(cs_T);
        printf("tensor fnorm before ALS: %lf\n", t_fnorm);
    }

    printf("Start fast aysm ALS, dim = %d, rank = %d, B = %d, b = %d\n", dim, rank, cs_T->B, cs_T->b);
    assert(cs_T->order == 3);
    
    double* XCB = new double[rank * dim];
    double* CC = new double[rank * rank];
    double* BB = new double[rank * rank];
        
    AsymCountSketch* cs_u1 = new AsymCountSketch(cs_T->hs[0]);
    AsymCountSketch* cs_u2 = new AsymCountSketch(cs_T->hs[1]);
    AsymCountSketch* cs_u3 = new AsymCountSketch(cs_T->hs[2]);
    
    cs_T->fft(fft_wrapper);
    
    for(int t = 0; t < T; t++) {
    
        //printf("Fast asym ALS round %d ...\n", t);
        
        fast_asym_ALS_update(cs_T, dim, rank, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, A, B, C, XCB, CC, BB, cs_u2, cs_u3);
        fast_asym_ALS_update(cs_T, dim, rank, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, B, C, A, XCB, CC, BB, cs_u2, cs_u3);
        fast_asym_ALS_update(cs_T, dim, rank, mat_wrapper, fft_wrapper, ifft_wrapper, lambda, C, A, B, XCB, CC, BB, cs_u2, cs_u3);
        
    }
    
    cs_T->fft(ifft_wrapper);
    
    delete[] XCB;
    delete[] CC;
    delete[] BB;
    
    if (report_residue) {

        for(int k = 0; k < rank; k++) {
            cs_u1->set_vector(A + k * dim, dim); cs_u1->fft(fft_wrapper);
            cs_u2->set_vector(B + k * dim, dim); cs_u2->fft(fft_wrapper);
            cs_u3->set_vector(C + k * dim, dim); cs_u3->fft(fft_wrapper);
            cs_T->add_rank_one_tensor(-lambda[k], dim, cs_u1, cs_u2, cs_u3, fft_wrapper, ifft_wrapper);
        }
        residue = fast_sqr_fnorm(cs_T);
        printf("After ALS: residue = %lf\n", residue);
        
        delete cs_u1;
        delete cs_u2;
        delete cs_u3;        
        
        return residue / t_fnorm;
    }
    else {
    
        delete cs_u1;
        delete cs_u2;
        delete cs_u3;    
    
        return 0;
    }

}

double fast_asym_tensor_power_method(AsymCountSketch* cs_T, int dim, int rank, int L, int T, Matlab_wrapper* mat_wrapper, FFT_wrapper* fft_wrapper, FFT_wrapper* ifft_wrapper, double* lambda, double** v, bool report_residue = false) {

    double residue = 0;
    double t_fnorm = 0;
    
    double* best_u = new double[dim];
    double* u = new double[dim];
    AsymCountSketch* asym_u[3];
    for(int i = 0; i < 3; i++) {
        asym_u[i] = new AsymCountSketch(cs_T->hs[i]);
    }
    
    if (report_residue) {
        t_fnorm = fast_sqr_fnorm(cs_T);
        printf("tensor fnorm before tensor power method: %lf\n", t_fnorm);
    }
    
    cs_T->fft(fft_wrapper);
    
    for(int k = 0; k < rank; k++) {
    
        printf("Round %d: ", k);
    
        double best_value = -1e100;
        memset(best_u, 0, sizeof(double) * dim);
        for(int tau = 0; tau < L; tau++) {
            putchar('.');
            fflush(stdout);
            generate_uniform_sphere_point(dim, u);
            for(int t = 0; t < T; t++) {
                
                asym_u[1]->set_vector(u, dim); asym_u[1]->fft(fft_wrapper);
                asym_u[2]->set_vector(u, dim); asym_u[2]->fft(fft_wrapper);
                
                fast_TIuv(cs_T, asym_u[1], asym_u[2], dim, fft_wrapper, ifft_wrapper, u);
                
                double sum = 0;
                for(int i = 0; i < dim; i++)
                    sum += SQR(u[i]);
                double scale = 1.0 / sqrt(sum);
                for(int i = 0; i < dim; i++)
                    u[i] *= scale;                
                                   
            }
            
            for(int i = 0; i < 3; i++) {
                asym_u[i]->set_vector(u, dim);
                asym_u[i]->fft(fft_wrapper);
            }
            double value = fast_Tuvw(cs_T, asym_u[0], asym_u[1], asym_u[2], dim, fft_wrapper, ifft_wrapper);
            if (value > best_value) {
                best_value = value;
                memcpy(v[k], u, sizeof(double) * dim);
            }
        }
        
        memcpy(u, v[k], sizeof(double) * dim);
        for(int t = 0; t < T; t++) {
        
            asym_u[1]->set_vector(u, dim); asym_u[1]->fft(fft_wrapper);
            asym_u[2]->set_vector(u, dim); asym_u[2]->fft(fft_wrapper);
              
            fast_TIuv(cs_T, asym_u[1], asym_u[2], dim, fft_wrapper, ifft_wrapper, u);
                
            double sum = 0;
            for(int i = 0; i < dim; i++)
                sum += SQR(u[i]);
            double scale = 1.0 / sqrt(sum);
            for(int i = 0; i < dim; i++)
                u[i] *= scale;                
                                                       
        }
        
        for(int i = 0; i < 3; i++) {
            asym_u[i]->set_vector(u, dim);
            asym_u[i]->fft(fft_wrapper);
        }   
        lambda[k] = fast_Tuvw(cs_T, asym_u[0], asym_u[1], asym_u[2], dim, fft_wrapper, ifft_wrapper);
        memcpy(v[k], u, sizeof(double) * dim);
        
        // deflation
        cs_T->add_rank_one_tensor(-lambda[k], dim, asym_u[0], asym_u[1], asym_u[2], fft_wrapper, ifft_wrapper, false);
        
        printf("#\n");
    
    }
    
    cs_T->fft(ifft_wrapper);
    if (report_residue) {
        residue = fast_sqr_fnorm(cs_T);
        printf("tensor fnorm after tensor power method: %lf\n", residue);
        residue /= t_fnorm;
    }
    
    delete best_u;
    delete u;
    for(int i = 0; i < 3; i++)
        delete asym_u[i];
        
    return residue;

}
