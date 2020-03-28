#!/bin/bash

#DIRS="$*"
GLOBAL_GRAPH_ID=0

function process_directory()
{
    local DIR=$1
    local OUTPUT_FILE=$2
    local TMPFILE=`mktemp`
    find ${DIR} -type f > ${TMPFILE}
    TMPFILE_FD=100

    echo "processing dir ${DIR}"

    exec 100<${TMPFILE}
    while read -u 100 line 
    do
        echo "t # ${GLOBAL_GRAPH_ID}" >> ${OUTPUT_FILE}
        cat ${line} >> ${OUTPUT_FILE}
        ((GLOBAL_GRAPH_ID++))
    done

    echo ${GLOBAL_GRAPH_ID}

    rm ${TMPFILE}
}


function process_all_subdirectories()
{
    local OUTPUTFILE=$1
    local DATADIR=../../graph_fam
    local TMPFILE=`mktemp`
    local TMPFILE_FD=10

    find ${DATADIR} -type d | tail -n +2 > ${TMPFILE}

    rm -rf ${OUTPUTFILE}
    touch ${OUTPUTFILE}

    exec 10<>${TMPFILE}
    while read -u 10 line 
    do
        echo "line: '${line}'"
        process_directory $line ${OUTPUTFILE}
    done

    rm ${TMPFILE}
}


process_all_subdirectories xxx.dat


