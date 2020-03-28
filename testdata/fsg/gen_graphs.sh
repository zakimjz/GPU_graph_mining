#!/bin/bash

GGEN=./ggen

function create_gdb()
{
    TRAN_NO=$1
    VERTEX_LABELS=$2
    EDGE_LABELS=$3
    PAT_LEN=$4
    PATTERNS=$5
    TSIZE=$6

    DATA_FILE=T${TRAN_NO}E${EDGE_LABELS}PL${PAT_LEN}P${PATTERNS}TL${TSIZE}V${VERTEX_LABELS}
    echo "generating ${DATA_FILE}"
    ./ggen -D ${TRAN_NO} -E ${EDGE_LABELS} -i ${PAT_LEN} -L ${PATTERNS} -S -T ${TSIZE} -V ${VERTEX_LABELS} > ${DATA_FILE}.dat -s 342513
    ../../src/globals/convertdb -fsg ${DATA_FILE}.dat -cudabin ${DATA_FILE}.cudat
    #cat ${DATA_FILE}.tmp | grep '^#.*$' > ${DATA_FILE}.info
    #cat ${DATA_FILE}.tmp | grep -v '^#.*$' > ${DATA_FILE}.dat
    #rm ${DATA_FILE}.tmp
}



#create_gdb 100000 5 5 4 50 15
#create_gdb 100000 5 5 4 50 30

#create_gdb 1000000 5 5 4 50 30

create_gdb $((1000 * 1000)) 5 5 4 50 30
create_gdb $((750 * 1000)) 5 5 4 50 30
create_gdb $((500 * 1000)) 5 5 4 50 30
create_gdb $((400 * 1000)) 5 5 4 50 30
create_gdb $((300 * 1000)) 5 5 4 50 30
create_gdb $((200 * 1000)) 5 5 4 50 30
create_gdb $((100 * 1000)) 5 5 4 50 30


