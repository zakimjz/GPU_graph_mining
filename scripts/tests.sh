#!/bin/bash



. ./create_par_makefile.sh
. ./create_seq_makefile.sh

SUPPORT_LIST="10 20 30"
INPUT_GRAPHDB="Chemical_340"
FILETYPE="-txt"
MAKEFILE="`get_seq_makefile_name ${INPUT_GRAPHDB}`"
get_seq_makefile_targets ${INPUT_GRAPHDB} ${SUPPORT_LIST} > ${SCRIPTDIR}/${MAKEFILE}
get_seq_makefile_content ${INPUT_GRAPHDB} ${FILETYPE} ${SUPPORT_LIST}  >> ${SCRIPTDIR}/${MAKEFILE}

#SUPPORT_LIST="30 40"
#SUPPORT_LIST="100 120 130 150 170 180 190"
#INPUT_GRAPHDB="Chemical_340"
#INPUT_GRAPHDB="fsg/T100000E5PL4P50TL15V5.dat"

#SUPPORT_LIST="10000 20000 30000 40000 50000 100000"
#SEQ_INPUT_GRAPHDB="fsg/T1000000E5PL4P50TL30V5.dat"
#PAR_INPUT_GRAPHDB="T1000000E5PL4P50TL30V5.cudabin"

SUPPORT_LIST="1000 2000 3000 4000 5000 10000"
SEQ_INPUT_GRAPHDB="fsg/T100000E5PL4P50TL30V5.dat"
PAR_INPUT_GRAPHDB="T100000E5PL4P50TL30V5.cudabin"


CONFIG_FILE="dummy_config"
CONFIG_ID="dummy_id"
SEQ_FILETYPE="-fsg"
PAR_FILETYPE="-cudabin"

export LOG_LEVEL=INFO

## Sequential gSpan 
MAKEFILE="`get_seq_makefile_name ${SEQ_INPUT_GRAPHDB}`"
SEQ_MAKEFILE=${MAKEFILE}
get_seq_makefile_targets ${SEQ_INPUT_GRAPHDB} ${SUPPORT_LIST} > ${SCRIPTDIR}/${MAKEFILE}
get_seq_makefile_content ${SEQ_INPUT_GRAPHDB} ${SEQ_FILETYPE} ${SUPPORT_LIST} >> ${SCRIPTDIR}/${MAKEFILE}

get_seq_makefile_targets ${SEQ_INPUT_GRAPHDB} ${SUPPORT_LIST} 
get_seq_makefile_content ${SEQ_INPUT_GRAPHDB} ${SEQ_FILETYPE} ${SUPPORT_LIST}

echo ${SEQ_MAKEFILE}
#make -j 10 -f ${SEQ_MAKEFILE}
make -f ${SEQ_MAKEFILE}

echo
echo "-------------------------------------------------------------------------------------"
echo 

## CUDA gSpan different versions
#for CUDA_EXECFILE in gspan_cuda gspan_cuda_no_sort gspan_cuda_lists#
#tested: gspan_cuda_no_sort  gspan_cuda_freq_mindfs gspan_cuda_no_sort_block  gspan_cuda_mult_block
#todo:  gspan_cuda_freq_mindfs
#gspan_cuda_mult_block
#gspan_cuda_no_sort_block
for CUDA_EXECFILE in gspan_cuda_mult_block ;
do
   MAKEFILE="`get_par_makefile_name ${PAR_INPUT_GRAPHDB} ${CONFIG_FILE} ${CONFIG_ID} `"
   PAR_MAKEFILE=${MAKEFILE}
   local RUNID=${CUDA_EXECFILE}
   get_par_makefile_targets ${PAR_INPUT_GRAPHDB} ${CUDA_EXECFILE}  ${SUPPORT_LIST} > ${SCRIPTDIR}/${MAKEFILE}
   get_par_makefile_content ${CUDA_EXECFILE} ${PAR_INPUT_GRAPHDB} ${RUNID}  ${PAR_FILETYPE} ${SUPPORT_LIST} >> ${SCRIPTDIR}/${MAKEFILE}


   echo ${PAR_MAKEFILE}
   make -f ${PAR_MAKEFILE}

   echo -e "\n\tGSPAN CUDA version: ${CUDA_EXECFILE}"
   echo -e "\tGraph DB (SEQ): ${SEQ_INPUT_GRAPHDB}"
   echo -e "\tGraph DB (PAR): ${PAR_INPUT_GRAPHDB}"
   echo -e "\n\tsupport\ttime (seq)\ttime (par)\tfreq graphs (seq)\tfreq graphs (par)\tsame?"

   ## Process logfiles and report timing   
   for support in ${SUPPORT_LIST}
   do
      PAR_LOGFILE="` get_par_log_filename $support ${PAR_INPUT_GRAPHDB} ${RUNID} `"

      SEQ_LOGFILE="` get_seq_log_filename $support ${SEQ_INPUT_GRAPHDB} `"
 
      SEQ_TOTAL_TIME=`awk -v var_name=total_time -f parse_log.awk ${SEQ_LOGFILE}`
      PAR_TOTAL_TIME=`awk -v var_name=total_time -f parse_log.awk ${PAR_LOGFILE}`
      SEQ_FREQ_GRAPHS=`awk -v var_name=total_frequent_graphs -f parse_log.awk ${SEQ_LOGFILE}`
      PAR_FREQ_GRAPHS=`awk -v var_name=total_frequent_graphs -f parse_log.awk ${PAR_LOGFILE}`
 
      EQUALS="N"
      if [ ${SEQ_FREQ_GRAPHS} = ${PAR_FREQ_GRAPHS} ]; then
       	EQUALS="Y"
      fi
       echo -e "\t$support\t${SEQ_TOTAL_TIME}\t${PAR_TOTAL_TIME}\t${SEQ_FREQ_GRAPHS}\t${PAR_FREQ_GRAPHS}\t${EQUALS}"

   done
   echo -e ""
done


