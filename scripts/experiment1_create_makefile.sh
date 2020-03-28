#!/bin/bash



. ./create_par_makefile.sh
. ./create_seq_makefile.sh

#../testdata/fsg/T100000E5PL4P50TL30V5.dat
SUPPORT_LIST="10000 5000 4000 3000"
INPUT_GRAPHDB="T100000E5PL4P50TL30V5.dat"
CONFIG_FILE="config_T100000E5PL4P50TL30V5.cfg"
CONFIG_ID="test"
FILETYPE="-fsg"

MAKEFILE="`get_par_makefile_name ${INPUT_GRAPHDB} ${CONFIG_FILE} ${CONFIG_ID}`"
PAR_MAKEFILE=${MAKEFILE}
get_par_makefile_targets ${INPUT_GRAPHDB} ${CONFIG_FILE} ${CONFIG_ID}  ${SUPPORT_LIST} > ${SCRIPTDIR}/${MAKEFILE}
get_par_makefile_content ${INPUT_GRAPHDB} ${CONFIG_FILE} ${CONFIG_ID}  ${FILETYPE} ${SUPPORT_LIST} >> ${SCRIPTDIR}/${MAKEFILE}


MAKEFILE="`get_seq_makefile_name ${INPUT_GRAPHDB}`"
SEQ_MAKEFILE=${MAKEFILE}
get_seq_makefile_targets ${INPUT_GRAPHDB} ${SUPPORT_LIST} > ${SCRIPTDIR}/${MAKEFILE}
get_seq_makefile_content ${INPUT_GRAPHDB} ${FILETYPE} ${SUPPORT_LIST} >> ${SCRIPTDIR}/${MAKEFILE}


