#SILE:=-s
SILE:=-s
all: 
	@make $(SILE) -C globals $@ -j20
	@make $(SILE) -C cuda_pp $@ -j20
	@make $(SILE) -C reduction $@ -j20
	@make $(SILE) -C gspan $@ -j20
	@make $(SILE) -C gspan_cuda $@ -j20

deps: 
	@make $(SILE) -C globals $@ -j20
	@make $(SILE) -C cuda_pp $@ -j20
	@make $(SILE) -C reduction   $@ -j20
	@make $(SILE) -C gspan   $@ -j20
	@make $(SILE) -C gspan_cuda $@ -j20


clean: 
	@make $(SILE) -C globals $@
	@make $(SILE) -C cuda_pp $@
	@make $(SILE) -C reduction $@
	@make $(SILE) -C gspan $@
	@make $(SILE) -C gspan_cuda $@
