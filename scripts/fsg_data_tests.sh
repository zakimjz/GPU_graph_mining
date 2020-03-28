. ./create_par_makefile.sh
. ./create_seq_makefile.sh

function run_seq_test()
{
    local FILETYPE=$1
    local FILENAME=$2
    local RUNID=$3
    shift 3;
    local SUPPORTS=$*

    MAKEFILE="`get_seq_makefile_name ${FILENAME}`"
    SEQ_MAKEFILE=${MAKEFILE}
    get_seq_makefile_targets ${FILENAME} ${SUPPORTS} > ${SCRIPTDIR}/${MAKEFILE}
    get_seq_makefile_content ${FILENAME} ${FILETYPE} ${SUPPORTS} >> ${SCRIPTDIR}/${MAKEFILE}


    echo ${SEQ_MAKEFILE}
    make -f ${SEQ_MAKEFILE}
}


function run_cuda_test()
{
    local FILETYPE=$1
    local FILENAME=$2
    local RUNID=$3
    shift 3;
    local SUPPORTS="$@"
    echo "filetype: ${FILETYPE}" 
    echo "filename: ${FILENAME}"
    echo "supports: ${SUPPORTS}"

    MAKEFILE="`get_par_makefile_name ${FILENAME} ${RUNID} `"
    PAR_MAKEFILE=${MAKEFILE}
    get_par_makefile_targets ${FILENAME} ${RUNID}  ${SUPPORTS} > ${SCRIPTDIR}/${MAKEFILE}
    echo ${FILENAME}
    CUDA_EXECFILE=${RUNID}
    get_par_makefile_content ${CUDA_EXECFILE} ${FILENAME} ${RUNID}  ${FILETYPE} ${SUPPORTS} >> ${SCRIPTDIR}/${MAKEFILE}

    echo ${PAR_MAKEFILE}
    make -f ${PAR_MAKEFILE}
}




function run_all_seq_tests() {
    local FILENAME=$1
    local RUNID=$2
    shift 2;
    local SUPPORTS="$*"

    echo "run_all_tests: ${SUPPORTS}"
    echo "filename: ${FILENAME}" 
    echo "runid: ${RUNID}"
    SEQ_FILENAME=${FILENAME}.cudat
    run_seq_test -cudabin ${SEQ_FILENAME} ${RUNID} ${SUPPORTS}

#    CUDA_FILENAME=${FILENAME}.cudat
#    run_cuda_test -cudabin ${CUDA_FILENAME} ${RUNID} ${SUPPORTS}

    echo
    echo

}


function run_all_tests() {
    local FILENAME=$1
    local RUNID=$2
    shift 2;
    local SUPPORTS="$*"

    echo "run_all_tests: ${SUPPORTS}"
    echo "filename: ${FILENAME}" 
    echo "runid: ${RUNID}"
#    SEQ_FILENAME=${FILENAME}.cudat
#    run_seq_test -cudabin ${SEQ_FILENAME} ${RUNID} ${SUPPORTS}

    CUDA_FILENAME=${FILENAME}.cudat
    run_cuda_test -cudabin ${CUDA_FILENAME} ${RUNID} ${SUPPORTS}

    echo
    echo

}


function run_pdb_tests() {
    local FILENAME=$1
    local RUNID=$2
    shift 2;
    local SUPPORTS=$*

#    echo "executing pdb test on ${FILENAME}.dat"
#    SEQ_FILENAME=${FILENAME}.dat
#    run_seq_test -txt ${SEQ_FILENAME} ${RUNID} ${SUPPORTS}

    CUDA_FILENAME=${FILENAME}.cudat
    run_cuda_test -cudabin ${CUDA_FILENAME} ${RUNID} ${SUPPORTS}

    echo
    echo

}


#CUDA_EXECFILE=gspan_cuda_mult_block

binary=gspan_cuda_mult_block
run_all_seq_tests T100000E5PL4P50TL30V5 ${binary} 10000   5000  4000   3000  2000  1000 
run_all_seq_tests T200000E5PL4P50TL30V5 ${binary} 20000   10000  8000   6000  4000  2000
run_all_seq_tests T300000E5PL4P50TL30V5 ${binary} 30000   15000 12000   9000  6000  3000
run_all_seq_tests T400000E5PL4P50TL30V5 ${binary} 40000   20000 16000  12000  8000  4000
run_all_seq_tests T500000E5PL4P50TL30V5 ${binary} 50000   25000 20000  15000 10000  5000
run_all_seq_tests T750000E5PL4P50TL30V5 ${binary} 75000   37500 30000  25000 15000 75000
run_all_seq_tests T1000000E5PL4P50TL30V5 ${binary} 100000 50000 40000  30000 20000 10000




for binary in gspan_cuda_no_sort gspan_cuda_no_sort_block gspan_cuda_mult_block ; do
    echo "=============================================================================================================="
    echo "binary: ${binary}"
    run_all_tests T100000E5PL4P50TL30V5 ${binary} 10000    5000  4000   3000  2000  1000 
    run_all_tests T200000E5PL4P50TL30V5 ${binary} 20000   10000  8000   6000  4000  2000
    run_all_tests T300000E5PL4P50TL30V5 ${binary} 30000   15000 12000   9000  6000  3000
    run_all_tests T400000E5PL4P50TL30V5 ${binary} 40000   20000 16000  12000  8000  4000
    run_all_tests T500000E5PL4P50TL30V5 ${binary} 50000   25000 20000  15000 10000  5000
    run_all_tests T750000E5PL4P50TL30V5 ${binary} 75000   37500 30000  25000 15000 75000
    run_all_tests T1000000E5PL4P50TL30V5 ${binary} 100000 50000 40000  30000 20000 10000
done


#ulimit -t $((60*60*10))
for binary in gspan_cuda_no_sort gspan_cuda_no_sort_block gspan_cuda_mult_block ; do
    run_pdb_tests pdb_graphs ${binary} 1700 1600 1500 1400 1300 1200 1100 1000
done



echo "running NCI dataset"
#run_all_tests nci_v1 ${binary} 7000 8000 9000 10000 

for binary in gspan_cuda_no_sort gspan_cuda_no_sort_block gspan_cuda_mult_block ; do
#binary=seq
    run_all_tests nci_v1 ${binary} 7000 8000 9000 10000 12000 14000 16000 18000 20000 24000
    run_all_tests nci_v1 ${binary} 7000
done
