DEPSDIR:=deps
OBJSDIR:=objs

MPICXX:=mpicxx
CXX:=nvcc
DEPCXX:=g++
LD:=nvcc

#GSL_LIB:=-lgsl -lgslcblas



ifeq ($(shell uname -n), badr)

CUDAFLAGS:=-gencode arch=compute_20,code=sm_20 $(CUDA_DEFS) -x cu $(OPT_FLAGS)
#CXXFLAGS:=-g -O0 -DSTLPORT_UNAVAL -DLOG_TRACE $(CUDAFLAGS)
CXXFLAGS:=-O4 -DLOG_INFO -DSTLPORT_UNAVAL -m64 $(CUDAFLAGS)
DEPSCXXFLAGS:= -DLOG_INFO
INCLUDES:=-I. -I$(ROOT)/src/globals -I$(ROOT)/gtest/distr_nostlp/include
LDFLAGS:=

DEPINCLUDES:=-I/usr/local/cuda/include/

GTEST_LIB:=$(ROOT)/gtest/distr_nostlp/lib/libgtest.a
GTESTMAIN_LIB:=$(ROOT)/gtest/distr_nostlp/lib/libgtest_main.a
GTEST_LD:=$(GTEST_LIB) $(GTESTMAIN_LIB)
GTEST_INCLUDE:=-I$(ROOT)/gtest/distr_nostlp/include

endif


