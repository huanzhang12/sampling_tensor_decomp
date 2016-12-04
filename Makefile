CXXFLAGS= -march=native -O2 -g -Wno-unused-result -Wno-write-strings
SOURCES = $(wildcard *.cc)
HEADS = $(wildcard *.h)
OBJS := $(SOURCES:.cc=.o)
MAT_INC = /usr/local/MATLAB/R2014b/extern/include/
MAT_LIB = /usr/local/MATLAB/R2014b/bin/glnxa64

all: fftspec fftlda

%.o: %.cc $(HEADS)
	g++ $(CXXFLAGS) -I $(MAT_INC) -c $<

fftspec: $(OBJS)
	g++ $(CXXFLAGS) -L $(MAT_LIB) -o fftspec fftspec.o fast_tensor_power_method.o matlab_wrapper.o count_sketch.o tensor.o hash.o util.o -lfftw3 -leng -lmat -lmex -lmx -lut -lpthread -Wl,-rpath,$(MAT_LIB)

fftlda: $(OBJS)
	g++ $(CXXFLAGS) -L $(MAT_LIB) -o fftlda fftlda.o fast_tensor_power_method.o tensor_lda.o tensor_lda_multithread.o corpus.o LDA.o matlab_wrapper.o count_sketch.o tensor.o hash.o util.o -lfftw3 -leng -lmat -lmex -lmx -lut -lpthread -Wl,-rpath,$(MAT_LIB)

test: matlab_wrapper.o util.o
	g++ $(CXXFLAGS) -L $(MAT_LIB) -o test matlab_wrapper.o util.o -lfftw3 -leng -lmat -lmex -lmx -lut -lpthread -Wl,-rpath,$(MAT_LIB)

clean:
	rm -f *.o fftspec fftlda test

