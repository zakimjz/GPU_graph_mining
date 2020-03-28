ROOT:=../..
-include $(ROOT)/make/make_defs.mk

ALL_TARGETS:=libglobals.a globals_tests convertdb
INCLUDES:=-I. -I../cuda_pp $(GTEST_INCLUDE)

all: prepare $(ALL_TARGETS)


TARGET:=libglobals.a
SRCS:=cuda_graph_types.cpp \
      dbio.cpp \
      dfs_code.cpp \
      dfs_ismin.cpp \
      graph_getmin.cpp \
      graph_output.cpp \
      graph_types.cpp \
      gsl_rnd_gen.cpp \
      logger.cpp \
      utils.cpp \
      test_support.cpp \
      embedding_lists.cpp \
      memory_checker.cpp \
      graph_is_subgraph.cpp \
      check_database.cpp \
      fix_database.cpp
ALL_DEPS:=$(ALL_DEPS) $(SRCS:.cpp=.dep)
-include $(ROOT)/make/compile_lib.mk


TARGET:=globals_tests
SRCS:=convert_to_device_test.cpp dbio_test.cpp dfs_code_tests.cpp dfs_ismin_tests.cpp exception_test.cpp graph_get_min_dfs_code_test.cpp graph_is_subgraph_test.cpp serialization_test.cpp test_support.cpp fix_graph_test.cpp
ALL_DEPS:=$(ALL_DEPS) $(SRCS:.cpp=.dep)
LDFLAGS:=$(GTEST_LD) libglobals.a
TARGET_DEPS:=./libglobals.a
-include $(ROOT)/make/compile_exe.mk




TARGET:=convertdb
SRCS:=convertdb.cpp check_database.cpp
ALL_DEPS:=$(ALL_DEPS) $(SRCS:.cpp=.dep)
LDFLAGS:=$(GTEST_LD) libglobals.a
TARGET_DEPS:=./libglobals.a
-include $(ROOT)/make/compile_exe.mk


-include $(ROOT)/make/make_tools.mk

-include $(addprefix deps/,$(ALL_DEPS))

