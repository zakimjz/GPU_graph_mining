ROOT:=../../
-include $(ROOT)/make/make_defs.mk

ALL_TARGETS:=prepare gspan_seq libgspanseq.a

INCLUDES:=-I. -I../cuda_pp $(GTEST_INCLUDE) -I../globals

all: $(ALL_TARGETS)


TARGET:=gspan_seq
SRCS:=gspan.cpp main.cpp ismin.cpp
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a ../cuda_pp/libcudapp.a
ALL_DEPS:=$(ALL_DEPS) $(SRCS:.cpp=.dep)
-include $(ROOT)/make/compile_exe.mk



TARGET:=libgspanseq.a
SRCS:=gspan.cpp ismin.cpp
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a ../cuda_pp/libcudapp.a
ALL_DEPS:=$(ALL_DEPS) $(SRCS:.cpp=.dep)
-include $(ROOT)/make/compile_lib.mk




-include $(ROOT)/make/make_tools.mk
-include $(addprefix deps/,$(ALL_DEPS))


