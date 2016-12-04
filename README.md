Sublinear Time Tensor Decomposition with Importance Sampling
==================================

This repository contains experiment code for our paper, 
"Sublinear Time Orthogonal Tensor Decomposition", by Zhao Song, David P. Woodruff and Huan Zhang.
In our paper, we developed a sublinear algorithm for decomposing 
large tensors using robust tensor power method, where the
computationally expensive tensor contractions are done by importance
sampling based on the magnitude of the power iterate variable.

In our experiments we only implemented 3-rd order tensors, and 
sampling with or without pre-scanning variants of our algorithm.
Most parts of the code is based on [1], and we only add new importance
sampling based tensor contraction functions.

[Spotlight Video for this paper](https://vimeo.com/192582848)
[Poster for this paper](resources/poster.pdf)

Build
--------------------

We require the following environment to build the code:

- libfftw3 (provided by package `libfftw3-dev` on Debian based systems)
- Matlab 2014b or later
- GNU Compiler Collection (GCC) 4.9 or newer versions recommended
- Tested on Ubuntu 16.04, should be able to run other Linux distributions or OSX

You need to modify `Makefile` to make sure Matlab includes and libraries 
folders match your Matlab installation. They are defined as
variable `MAT_INC` and `MAT_LIB`:

```
MAT_INC = /usr/local/MATLAB/R2014b/extern/include
MAT_LIB = /usr/local/MATLAB/R2014b/bin/glnxa64

```

To build the program, simply run `make`. Two binaries, `fftspec`
and `fft-lda` will be built. 
We only need `fftspec` for experiments.

Datasets
---------------------------------

Synthetic dense tensors with different eigengaps and noises can be generated via `fftspec`.
The following command generates a rank-100, dimension 600\*600\*600 tensor, with sigma=0.01 noise added.
Generated tensors will be stored under the folder "data".

```
./fftspec synth_lowrank 600 100 0.01 1
```

The last parameter `1` indicates that the 100 eigenvalues decay as $\lambda_i = 1/i$.
If you set it to `2`, then $\lambda_i = 1/i^2$.
If you set it to `3`,then $\lambda_i = 1 - (i-1)/k$, where k is the rank.

For the tensors from LDA (Latent Dirichlet Allocation), we provide preprocessed tensors
(lzma compressed) in folder `LDA_tensor_200` for all 6 datasets we used.

Run Tensor Decomposition
----------------------------------

4 algorithms were provided for tensor decomposition:

* Naive robust tensor power method, computing exact tensor contractions
* Sketching based tensor power method, based on [1]
* Importance Sampling based tensor power method, without pre-scanning
* Importance Sampling based tensor power method, with pre-scanning

Importance Sampling with pre-scanning provides best theoretical bound where
only O(n^2) samples are needed, but it needs to scan the entire tensor first.
In practice, the without pre-scanning version (assuming the tensor has bounded
slice norms, see our paper for details) works better.

In sketching or sampling based algorithms, the following parameters are 
needed:

* T: Number of power iterations
* L: The number of starting vectors of the robust tensor power method
* B: The number of sketches used in sketching, or the number of repetitions of sampling 
* b: The size of the sketch, or the number of indices sampled

Note that there are small notation differences for the letter b between the code and paper:
for the sketching based method, the actual sketch length used is 2^b; and for 
importance sampling based method, the actual total number of samples is O(nb).

The following command shows how to run tensor decomposition with `fftspec`:

```
./fftspec algorithm input rank L T B b output
```

`rank` is the number of eigenpairs to be recoveried, 
`algorithm` can be `slow_rbp`, `fast_rbp`, `fast_sample_rbp` and `prescan_sample_rbp`, which
corresponds to the 4 algorithms metioned above, respectively.
For naive tensor power method `slow_rbp`, parameter B and b are not needed.

The following example shows how to generate a synthetic tensor, and run different methods
to compare their running time and recovery accuracy:

```
# generate a 800x800x800, rank-100 tensor (takes a while...)
./fftspec synth_lowrank 800 100 0.01 2
# Run sketching based tensor power method
./fftspec fast_rbp data/tensor_dim_800_rank_100_noise_0.01_decaymethod_2.dat 1 50 30 30 16 output_dim_800_rank_100_noise_0.01_decaymethod_2_fastrbp_50_30_30_16_rank1.dat
# Run sampling based tensor power method, without prescanning
./fftspec fast_sample_rbp data/tensor_dim_800_rank_100_noise_0.01_decaymethod_2.dat 1 50 30 30 10 output_dim_800_rank_100_noise_0.01_decaymethod_2_samplerbp_50_30_30_10_rank1.dat
# Run sampling based tensor power method, with prescanning
./fftspec prescan_sample_rbp data/tensor_dim_800_rank_100_noise_0.01_decaymethod_2.dat 1 50 30 30 10 output_dim_800_rank_100_noise_0.01_decaymethod_2_presamplerbp_50_30_30_10_rank1.dat
# Run naive tensor power method (SLOW!)
./fftspec slow_rbp data/tensor_dim_800_rank_100_noise_0.01_decaymethod_2.dat 1 50 30 output_dim_800_rank_100_noise_0.01_decaymethod_2_slowrbp_50_30_rank1.dat
# 
```

We want the residual norm to be small, while keeping the reported CPU time as
short as possible.  Fixing T and L, you can try different B and b and see how
running time and residual change.  Generally, using a smaller B and b makes the
algorithm run faster, but the residual is likely to increase and divergence may
occur when B or b is too small.  

For the example above here is the expected
outputs. If you got numbers quite different from these you probably hit a bug.

* For sketching based tensor power method:
```
# [STAT]: prep_cpu_time=12.654545
# [STAT]: prep_wall_time=12.654588
# [STAT]: cpu_time=69.057999
# [STAT]: wall_time=69.058114
# [STAT]: residue=0.089962
# [STAT]: fnorm=1.010029
```

* For importance sampling based tensor power method (without pre-scanning, no preprocessing time):
```
# [STAT]: cpu_time=9.078225
# [STAT]: wall_time=9.078256
# [STAT]: residue=0.086472
# [STAT]: fnorm=1.010029

```

* For importance sampling based tensor power method (with pre-scanning):
```
# [STAT]: prep_cpu_time=0.659961
# [STAT]: prep_wall_time=0.659971
# [STAT]: cpu_time=10.123327
# [STAT]: wall_time=10.123355
# [STAT]: residue=0.086454
# [STAT]: fnorm=1.010029

```

* For naive tensor power method:
```
Too long, don't run!
```

Results
----------------------------------

We have included a large range of results in the `results` folder, for both sketching and importance sampling
based parameters with different B and b, and different eigenvalue decay rates. 

Contact information
----------------------------------

If your have any questions or comments, please open an issue on Github,
or send an email to ecezhang@ucdavis.edu. We appreciate your feedback.

References
----------------------------------

[1] Yining Wang, Hsiao-Yu Tung, Alex J Smola, and Anima Anandkumar. Fast and
guaranteed tensor decomposition via sketching. In NIPS, pages 991-999, 2015.

