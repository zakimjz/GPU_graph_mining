#! /bin/bash

. ./local_vars.sh

#AWK_FILE="${SCRIPTDIR}/parse_log.awk"

INPUT_GRAPHDB="Chemical_340"
CONFIG_FILE="dummy_config"
CONFIG_ID="dummy_id"
SUPPORT_LIST="10 20"

echo -e "\tGraph DB: ${INPUT_GRAPHDB}"
echo -e "\tsupport\ttime (seq)\ttime (par)\tfreq graphs (seq)\tfreq graphs (par)\tsame?"
for support in ${SUPPORT_LIST}
do
 PAR_LOGFILE="` get_par_log_filename $support ${INPUT_GRAPHDB} ${CONFIG_FILE} ${CONFIG_ID} `"

 SEQ_LOGFILE="` get_seq_log_filename $support ${INPUT_GRAPHDB} `"
 
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
