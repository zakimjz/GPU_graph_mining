ROOT:=../../
-include $(ROOT)/make/make_defs.mk

ALL_TARGETS:=prepare libreduction.a reduction_test scan_test copy_test all_tests

INCLUDES:=-I. -I../cuda_pp $(GTEST_INCLUDE) -I../globals -I../gspan -I ../cudpp

all: $(ALL_TARGETS)


TARGET:=libreduction.a
SRCS:=mgpuscan.cpp cuda_segmented_op.cpp cuda_segmented_reduction.cpp cucpp.cpp cuda_code.cpp
ALL_DEPS:=$(SRCS:.cpp=.dep)
-include $(ROOT)/make/compile_lib.mk




TARGET:=reduction_test
SRCS:=reduction_test.cpp reduction_test_bugs.cpp
ALL_DEPS:=$(SRCS:.cpp=.dep)
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a libreduction.a ../cuda_pp/libcudapp.a
LD_DEPS:=libreduction.a ../cuda_pp/libcudapp.a
-include $(ROOT)/make/compile_exe.mk




TARGET:=scan_test
SRCS:=scan_test.cpp
ALL_DEPS:=$(SRCS:.cpp=.dep)
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a libreduction.a ../cuda_pp/libcudapp.a
LD_DEPS:=libreduction.a ../cuda_pp/libcudapp.a
-include $(ROOT)/make/compile_exe.mk



TARGET:=copy_test
SRCS:=copy_test.cpp
ALL_DEPS:=$(SRCS:.cpp=.dep)
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a libreduction.a ../cuda_pp/libcudapp.a
LD_DEPS:=libreduction.a ../cuda_pp/libcudapp.a
-include $(ROOT)/make/compile_exe.mk




TARGET:=all_tests
SRCS:=reduction_test.cpp reduction_test_bugs.cpp reduction_test_bug2.cpp scan_test.cpp copy_test.cpp
ALL_DEPS:=$(SRCS:.cpp=.dep)
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a libreduction.a ../cuda_pp/libcudapp.a
LD_DEPS:=libreduction.a ../cuda_pp/libcudapp.a
-include $(ROOT)/make/compile_exe.mk






-include $(ROOT)/make/make_tools.mk
-include $(addprefix deps/,$(ALL_DEPS))



