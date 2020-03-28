ROOT:=../../
-include $(ROOT)/make/make_defs.mk

ALL_TARGETS:=libcudapp.a cuda_configurator_test

all: prepare $(ALL_TARGETS)


TARGET:=libcudapp.a
SRCS:=cuda_configurator.cpp
ALL_DEPS:=$(SRCS:.cpp=.dep)
INCLUDES:=-I. -I../cuda_pp $(GTEST_INCLUDE) -I../globals
LDFLAGS:=$(GTEST_LD) 
TARGET_DEPS:=
-include $(ROOT)/make/compile_lib.mk



TARGET:=cuda_configurator_test
SRCS:=cuda_configurator_test.cpp
ALL_DEPS:=$(SRCS:.cpp=.dep)
INCLUDES:=-I. -I../cuda_pp $(GTEST_INCLUDE) -I../globals
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a ./libcudapp.a
TARGET_DEPS:=../globals/libglobals.a ./libcudapp.a
-include $(ROOT)/make/compile_exe.mk


-include $(ROOT)/make/make_tools.mk
-include $(addprefix deps/,$(ALL_DEPS))


