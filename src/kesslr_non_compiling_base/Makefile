SRC:=dbio.cpp  dfs_code.cpp  dfs_ismin.cpp graph_getmin.cpp  graph_output.cpp  graph_types.cpp  gsl_rnd_gen.cpp  logger.cpp  utils.cpp
AR:=ar
CXX:=g++
INCLUDES:=-I. -isystem /usr/include/stlport 
LDFLAGS:=-lstlport
CXXFLAGS:=-O0 -ggdb

all: prepare libkesslr.a

libkesslr.a: $(addprefix objs/,$(SRC:.cpp=.o))
	@echo AR $@
	@${AR} crs $@ $(SRC)

objs/%.o: %.cpp
	@echo CXX $<
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $^ -o $@  $(INCLUDES)

prepare:
	mkdir -p objs deps

clean:
	rm -rf libkesslr.a *.o *~ objs deps



