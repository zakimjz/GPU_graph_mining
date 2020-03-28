#!/bin/bash

export LOG_LEVEL=INFO

# run test
./gspan_cuda_test -txt ../../testdata/Chemical_340 10 xx xx

# run gspan_cuda on some database
#./gspan_cuda -txt ../../testdata/Chemical_340 200 d d

