#!/bin/bash

#http://cactus.nci.nih.gov/cgi-bin/nci2.1.tcl?output=sdf&op1=nsc&data1=191&nomsg=1

START=1
END=1000000
STEP=1000
OUTDIR=ncidata

function get_range()
{
    if [ ! -d ${OUTDIR} ] ; then
        mkdir -p ${OUTDIR}
    fi

    local OUTFILE=${OUTDIR}/$1;
    shift 1;
    local NSCIDXS="$*"

    wget -O ${OUTFILE} "http://cactus.nci.nih.gov/cgi-bin/nci2.1.tcl?output=sdf&op1=nsc&data1=${NSCIDXS}&nomsg=1"
}

exec 100<nscids

function get_nscids()
{
    local COUNT=$1
    local RESULT=""
    read -u 100 line

    RESULT="${line}"

    while read -u 100 line 
    do
        RESULT="${RESULT}+${line}"
        if [ "x${COUNT}" = "x0" ] ; then
            break;
        fi
        COUNT=$((COUNT-1))
#        echo "${COUNT} " >&2
    done
    echo -n ${RESULT} | tr -d '\r'
#sed 's/\r//g'
#tr -d '\n'
}

#for first in `seq $START $STEP $END` ; do
first=0
while true
do
    echo $f
    #RANGE=$( seq -s'+' ${first} $((first+STEP-1)) )
    RANGE=$(get_nscids ${STEP})
    OF="nci-${first}.sdf"
    echo ${RANGE}
    get_range ${OF} ${RANGE}
    if [ "x" = "x${RANGE}" ] ; then
        break;
    fi

#    break;
    first=$((first+${STEP}))

    #get_range ${OF} ${RANGE}
    #cat ${OUTDIR}/${OF} >> ncidata.sdf
done

