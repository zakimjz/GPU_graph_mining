#! /bin/bash

. ./local_vars.sh

export LOG_LEVEL=T1
./create_par_makefile.sh
./create_seq_makefile.sh

make -f makefile_par*
make -f makefile_seq*

./process_log.sh
