. ./create_par_makefile.sh
. ./create_seq_makefile.sh
. ./psql_support.sh

function postprocess_experiments() {
    local FILENAME=$1
    local RUNID=$2
    shift 2;
    local SUPPORTS=$*

    local SEQ_INPUT_GRAPHDB="${FILENAME}.dat"
    local PAR_INPUT_GRAPHDB="${FILENAME}.cudat"

    if [ "${STORE_IN_DB}" = "YES" ] ; then
	psql_insert_database ${FILENAME}
	psql_create_experiment ${FILENAME} ${RUNID}
    fi

    echo -e "\n\tGSPAN CUDA version: ${RUNID}"
    echo -e "\tGraph DB (SEQ): ${SEQ_INPUT_GRAPHDB}"
    echo -e "\tGraph DB (PAR): ${PAR_INPUT_GRAPHDB}"
    echo -e "\n\tsupport\ttime (seq)\ttime (par)\tfreq graphs (seq)\tfreq graphs (par)\tsame?"

    for support in ${SUPPORTS}
    do
        PAR_LOGFILE="` get_par_log_filename $support ${PAR_INPUT_GRAPHDB} ${RUNID} `"

        SEQ_LOGFILE="` get_seq_log_filename $support ${SEQ_INPUT_GRAPHDB} `"
 
        SEQ_TOTAL_TIME=`awk -v var_name=total_time -f parse_log.awk ${SEQ_LOGFILE}`
        PAR_TOTAL_TIME=`awk -v var_name=total_time -f parse_log.awk ${PAR_LOGFILE}`
        SEQ_FREQ_GRAPHS=`awk -v var_name=total_frequent_graphs -f parse_log.awk ${SEQ_LOGFILE}`
        PAR_FREQ_GRAPHS=`awk -v var_name=total_frequent_graphs -f parse_log.awk ${PAR_LOGFILE}`

        SPEEDUP=$(echo "${SEQ_TOTAL_TIME}/${PAR_TOTAL_TIME}" | bc -l | sed 's/\(\....\).*/\1/g')

        EQUALS="N"
#	echo "${SEQ_FREQ_GRAPHS}" = "${PAR_FREQ_GRAPHS}"
        if [ "${SEQ_FREQ_GRAPHS}" = "${PAR_FREQ_GRAPHS}" ]; then
       	    EQUALS="Y"
        fi
        echo -e "\t$support\t${SEQ_TOTAL_TIME}\t${PAR_TOTAL_TIME}\t${SEQ_FREQ_GRAPHS}\t${PAR_FREQ_GRAPHS}\t${EQUALS}\t${SPEEDUP}"

	if [ "${STORE_IN_DB}" = "YES" ] ; then
	    psql_process_experiment ${FILENAME} ${RUNID} ${support} ${SEQ_TOTAL_TIME} ${PAR_TOTAL_TIME}
	fi
    done
}



#DB_TIMESTAMP=$(date -u)
export DB_TIMESTAMP='Tue Feb  5 13:44:09 EST 2013'
export DESCRIPTION="This is the one of the first experiments performed on GPUs that gives slightly better speedups then 1."
export STORE_IN_DB="NO"

#CUDA_EXECFILE=gspan_cuda_no_sort
#CUDA_EXECFILE=gspan_cuda_mult_block

#if [ "a" = "b" ] ; then

for CUDA_EXECFILE in gspan_cuda_mult_block gspan_cuda_no_sort gspan_cuda_no_sort_block ; do

postprocess_experiments T100000E5PL4P50TL30V5 ${CUDA_EXECFILE} 10000    5000  4000   3000  2000  1000 
postprocess_experiments T200000E5PL4P50TL30V5 ${CUDA_EXECFILE} 20000   10000  8000   6000  4000  2000
postprocess_experiments T300000E5PL4P50TL30V5 ${CUDA_EXECFILE} 30000   15000 12000   9000  6000  3000
postprocess_experiments T400000E5PL4P50TL30V5 ${CUDA_EXECFILE} 40000   20000 16000  12000  8000  4000
postprocess_experiments T500000E5PL4P50TL30V5 ${CUDA_EXECFILE} 50000   25000 20000  15000 10000  5000
postprocess_experiments T750000E5PL4P50TL30V5 ${CUDA_EXECFILE} 75000   37500 30000  25000 15000 75000
postprocess_experiments T1000000E5PL4P50TL30V5 ${CUDA_EXECFILE} 100000 50000 40000  30000 20000 10000

done
#fi
exit;

for binary in gspan_cuda_no_sort gspan_cuda_no_sort_block gspan_cuda_mult_block ; do
    postprocess_experiments nci_v1 ${binary} 7000 8000 9000 10000 12000 14000 16000 18000 20000 24000
done
