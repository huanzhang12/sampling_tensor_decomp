#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "util.h"
#include "config.h"
#include "fftw3.h"

int compare_double(const void* p1, const void* p2) {
	double* dp1 = (double*)p1;
	double* dp2 = (double*)p2;
	if (*dp1 < *dp2) return -1;
	else if (*dp1 == *dp2) return 0;
	else return 1;
}

int compare_int(const void* p1, const void* p2) {
	int* dp1 = (int*)p1;
	int* dp2 = (int*)p2;
	if (*dp1 < *dp2) return -1;
	else if (*dp1 == *dp2) return 0;
	else return 1;
}

void print_complex_vector(const fftw_complex* a, int len) {
	for(int i = 0; i < len; i++)
		printf("%lf + %lf i\n", a[i][0], a[i][1]);
}

double generate_std_normal() {
	// Polar Box-Muller method
	double u = (double)rand() / RAND_MAX * 2 - 1;
	double v = (double)rand() / RAND_MAX * 2 - 1;
	double s = SQR(u) + SQR(v);
	while (s >= 1 || fabs(s) <= 1e-7) {
		u = (double)rand() / RAND_MAX * 2 - 1;
		v = (double)rand() / RAND_MAX * 2 - 1;
		s = SQR(u) + SQR(v);
	}
	return u * sqrt(-2 * log(s) / s);
}

void generate_uniform_sphere_point(int dim, double* ret) {

	int p = 0;

	// generate random gaussians
	while (p+1 < dim) {
		double u = (double)rand() / RAND_MAX * 2 - 1;
		double v = (double)rand() / RAND_MAX * 2 - 1;
		double s = SQR(u) + SQR(v);
		while (s >= 1 || fabs(s) <= 1e-7) {
			u = (double)rand() / RAND_MAX * 2 - 1;
			v = (double)rand() / RAND_MAX * 2 - 1;
			s = SQR(u) + SQR(v);
		}
		double t = sqrt(-2 * log(s) / s);
		ret[p] = u * t;
		ret[p+1] = v * t;
		p += 2;
	}
	if (dim & 1) ret[dim-1] = generate_std_normal();

	// normalize
	double norm = 0;
	for(int i = 0; i < dim; i++)
		norm += SQR(ret[i]);
	double scale = 1.0 / sqrt(norm);
	for(int i = 0; i < dim; i++)
		ret[i] *= scale;

}

void gram_schmidt_process(int n, int dim, double** x) {

	double* t = new double[dim];

	for(int i = 0; i < n; i++)  {
		double norm = 0;
		for(int j = 0; j < dim; j++) norm += SQR(x[i][j]);
		double scale = 1.0 / norm;
		memcpy(t, x[i], sizeof(double) * dim);
		for(int r = 0; r < i; r++) {
			double inner_prod = 0;
			for(int j = 0; j < dim; j++) inner_prod += x[i][j] * x[r][j];
			for(int j = 0; j < dim; j++)
				t[j] -= inner_prod * scale * x[r][j];
		}
		norm = 0;
		for(int j = 0; j < dim; j++) norm += SQR(t[j]);
		scale = 1.0 / sqrt(norm);
        
		for(int j = 0; j < dim; j++)
			x[i][j] = t[j] * scale;
	}

	delete[] t;

}

void sort_eigen_value_vector(double* lambda, double** v, int n, int dim) {

	double* t = new double[dim];	
	double t_scalar;

	for(int i = 0; i < n; i++)
		for(int j = i+1; j < n; j++)
			if (lambda[j] > lambda[i]) {
				t_scalar = lambda[i];
				lambda[i] = lambda[j];
				lambda[j] = t_scalar;
				memcpy(t, v[i], sizeof(double) * dim);
				memcpy(v[i], v[j], sizeof(double) * dim);
				memcpy(v[j], t, sizeof(double) * dim);
			}

	delete[] t;

}

int parse_cmd_arguments(int argc, char* argv[], int* para_groups, char** para_names, char** para_values) {

	int p = 0;

	int i = 0;
	while (i+1 < argc) {
		if (argv[i][0] == '-') {
			para_names[p] = new char[MAX_CMD_ARGUMENT_LEN];
			para_values[p] = new char[MAX_CMD_ARGUMENT_LEN];
			memset(para_names[p], 0, MAX_CMD_ARGUMENT_LEN);
			memset(para_values[p], 0, MAX_CMD_ARGUMENT_LEN);
			strcpy(para_names[p], argv[i]);
			strcpy(para_values[p], argv[i+1]);
			p ++;
			i += 2;
		}
		else {
			i ++;
		}
	}

	*para_groups = p;
	return p;

}

double gamma_sampling(double alpha) {
	
	// acceptance - rejection algorithm, http://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables

//	printf("alpha = %lf\n", alpha);
	
	double v0, v1, v2;
	double xi, eta;
	
	int k = floor(alpha);
	double delta = alpha - k;
	
	while (1) {
		v0 = (double)rand() / RAND_MAX;
		v1 = (double)rand() / RAND_MAX;
		v2 = (double)rand() / RAND_MAX;
		if (v2 <= exp(1) / (exp(1) + delta)) {
			xi = exp(1.0/delta * log(v1));
			eta = exp((delta-1) * log(xi)) * v0;
		}
		else {
			xi = 1 - log(v1);
			eta = v0 * exp(-xi);
		}
		if (eta <= exp((delta-1) * log(xi)) * exp(-xi))
			break;
	}
	
	
	double ret = 0;
	for(int i = 0; i < k; i++)
		ret += -log((double)(rand() + 1) / (RAND_MAX));

//	printf("ret = %lf, xi = %lf\n", ret, xi);
	
	return ret + xi;

}

void dirichlet_sampling(double* ret, double* p, int d) {

	/*for(int i = 0; i < d; i++)
		fprintf(stderr, "%lf ", p[i]);
	fprintf(stderr, "\n");*/

	for(int i = 0; i < d; i++)
		ret[i] = gamma_sampling(p[i]);
	double sum = 0;
	for(int i = 0; i < d; i++) sum += ret[i];
	for(int i = 0; i < d; i++)
		ret[i] /= sum;
	for(int i = 0; i < d; i++)
		ret[i] = (ret[i] + 1e-6) / (1.0 + 1e-6 * d);

	for(int i = 0; i < d; i++) {
		//fprintf(stderr, "%lf\n", ret[i]);
	    //printf("ret[%d] = %lf\n", i, ret[i]);
		if (ret[i] < 1e-9) {
			printf("ret[%d] = %lf\n", i, ret[i]);
		}
		assert(ret[i] >= 1e-9);
	}
		
}

int categorical_sampling(double* p, int d) {

    double sum = 0;
    for(int i = 0; i < d; i++) {
        assert(p[i] >= 0);
        sum += p[i];
    }
    assert(sum > 0);
    
    double t = (double)rand() / RAND_MAX * sum;
    int i = 0;
    double tsum = 0;
    while (i < d-1) {
        tsum += p[i];
        if (tsum >= t) break;
        i ++;
    }
    
    return i;

}
