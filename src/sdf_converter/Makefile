ROOT:=../../
-include $(ROOT)/make/make_defs.mk

ALL_TARGETS:=prepare sdfconvert

INCLUDES:=-I. $(GTEST_INCLUDE) -I../globals  -I/usr/include/openbabel-2.0

all: $(ALL_TARGETS)


TARGET:=sdfconvert
SRCS:=main.cpp
LDFLAGS:=../globals/libglobals.a -lopenbabel
LD_DEPS:=../globals/libglobals.a
-include $(ROOT)/make/compile_exe.mk




-include $(ROOT)/make/make_tools.mk
-include $(addprefix deps/,$(ALL_DEPS))


