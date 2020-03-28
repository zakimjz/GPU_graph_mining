ROOT:=../../
-include $(ROOT)/make/make_defs.mk

ALL_TARGETS:=prepare \
             libgspan_cuda.a \
             first_embed_test \
             ext_test \
             ext_test2 \
             gspan_cuda \
             gspan_cuda_lists \
             gspan_cuda_no_sort \
             gspan_cuda_no_sort_block \
             gspan_cuda_mult_block \
             gspan_cuda_mindfs \
             gspan_cuda_test \
             various_tests \
             fill_labels_test

INCLUDES:=-I. -I../cuda_pp $(GTEST_INCLUDE) -I../globals -I../gspan -I ../cudpp -I../reduction

all: $(ALL_TARGETS)


TARGET:=libgspan_cuda.a
SRCS:=compute_extensions.cpp \
      first_embedding.cpp \
      gspan_cuda.cpp \
      compute_support.cpp \
      extend_embeddings.cpp \
      test_embeddings.cpp \
      gspan_cuda_tests.cpp \
      filter_backward.cpp \
      filter_rmpath.cpp \
      cuda_utils.cpp \
      gspan_cuda.cpp \
      gspan_cuda_lists.cpp \
      gspan_cuda_no_sort.cpp \
      gspan_cuda_no_sort_block.cpp \
      gspan_cuda_mindfs.cpp \
      gspan_cuda_mult_block.cpp
ALL_DEPS:=$(ALL_DEPS) $(SRCS:.cpp=.dep)
-include $(ROOT)/make/compile_lib.mk


TARGET:=first_embed_test
SRCS:=first_embedding.cpp first_embed_test.cpp  test_embeddings.cpp
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a ../cuda_pp/libcudapp.a
LD_DEPS:=libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk


TARGET:=various_tests
SRCS:=sort_test.cpp \
	db_label_compaction_test.cpp \
	extract_extensions_test.cpp \
	filter_backward_embeddings_test.cpp \
	compute_support_remapped_test.cpp \
    compute_support_mult_block_test.cpp
#compute_extensions.cpp compute_support.cpp extend_embeddings.cpp ext_test.cpp filter_backward.cpp
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a ../cuda_pp/libcudapp.a ../reduction/libreduction.a libgspan_cuda.a
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a  ../cuda_pp/libcudapp.a libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk



TARGET:=ext_test
SRCS:=ext_test.cpp
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a ../cuda_pp/libcudapp.a ../reduction/libreduction.a libgspan_cuda.a
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=libgspan_cuda.a ../globals/libglobals.a ../cuda_pp/libcudapp.a 
-include $(ROOT)/make/compile_exe.mk


TARGET:=ext_test2
SRCS:=ext_test2.cpp
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a
LD_DEPS:=libgspan_cuda.a
TARGET_DEPS:=libgspan_cuda.a ../globals/libglobals.a ../cuda_pp/libcudapp.a 
-include $(ROOT)/make/compile_exe.mk


TARGET:=fill_labels_test
SRCS:=cuda_fill_labels_test.cpp
LDFLAGS:=$(GTEST_LD) ../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a ../reduction/libreduction.a
LD_DEPS:=libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk




TARGET:=gspan_cuda
SRCS:=main_gspan_cuda.cpp
LDFLAGS:=../globals/libglobals.a ../cuda_pp/libcudapp.a ../reduction/libreduction.a libgspan_cuda.a
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk



TARGET:=gspan_cuda_lists
SRCS:=main_gspan_cuda_lists.cpp
LDFLAGS:=../globals/libglobals.a ../cuda_pp/libcudapp.a ../reduction/libreduction.a libgspan_cuda.a
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk





TARGET:=gspan_cuda_no_sort
SRCS:=main_gspan_cuda_no_sort.cpp
LDFLAGS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a ../reduction/libreduction.a
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk



TARGET:=gspan_cuda_no_sort_block
SRCS:=gspan_cuda_no_sort_block_support.cpp common_main.cpp
LDFLAGS:=../globals/libglobals.a ../cuda_pp/libcudapp.a  libgspan_cuda.a ../reduction/libreduction.a
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk


TARGET:=gspan_cuda_mindfs
SRCS:=gspan_cuda_mindfs_support.cpp common_main.cpp
LDFLAGS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a ../reduction/libreduction.a 
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk



TARGET:=gspan_cuda_test
SRCS:=gspan_test.cpp
LDFLAGS:=../globals/libglobals.a ../cuda_pp/libcudapp.a ../gspan/libgspanseq.a ../reduction/libreduction.a libgspan_cuda.a  
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a ../cuda_pp/libcudapp.a
-include $(ROOT)/make/compile_exe.mk



TARGET:=gspan_cuda_freq_mindfs
SRCS:=common_main.cpp gspan_cuda_freq_mindfs_support.cpp
LDFLAGS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a ../reduction/libreduction.a
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk

TARGET:=gspan_cuda_mult_block
SRCS:=common_main.cpp gspan_cuda_mult_block_support.cpp
LDFLAGS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a ../reduction/libreduction.a
LD_DEPS:=libgspan_cuda.a ../reduction/libreduction.a
TARGET_DEPS:=../globals/libglobals.a ../cuda_pp/libcudapp.a libgspan_cuda.a
-include $(ROOT)/make/compile_exe.mk


-include $(ROOT)/make/make_tools.mk
-include $(addprefix deps/,$(ALL_DEPS))


