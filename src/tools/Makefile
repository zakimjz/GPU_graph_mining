ROOT:=../../
-include $(ROOT)/make/make_defs.mk

ALL_TARGETS:=prepare print_pattern_stats print_db_stats

INCLUDES:=-I. -I../cuda_pp $(GTEST_INCLUDE) -I../globals -I../gspan -I ../cudpp -I../reduction

all: $(ALL_TARGETS)


TARGET:=print_pattern_stats
SRCS:=print_pattern_stats.cpp
ALL_DEPS:=$(ALL_DEPS) $(SRCS:.cpp=.dep)
LDFLAGS:=../globals/libglobals.a
LD_DEPS:=../globals/libglobals.a
-include $(ROOT)/make/compile_exe.mk


TARGET:=print_db_stats
SRCS:=print_db_stats.cpp
ALL_DEPS:=$(ALL_DEPS) $(SRCS:.cpp=.dep)
LDFLAGS:=../globals/libglobals.a
LD_DEPS:=../globals/libglobals.a
-include $(ROOT)/make/compile_exe.mk


-include $(ROOT)/make/make_tools.mk
-include $(addprefix deps/,$(ALL_DEPS))




