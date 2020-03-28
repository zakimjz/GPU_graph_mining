#!/bin/bash


if [ "`uname -n`" = "socrates" ] ; then
    PROJ_DIR=/home/kesslr/projects/parallel-sub-graph-mining-on-gpu
fi

if [ "`uname -n`" = "badr" ] ; then

  if [ "`whoami`" = "kesslr" ]; then 
    PROJ_DIR=/mnt/d/home/kesslr/parallel-sub-graph-mining-on-gpu
  fi
 
  
  if [ "`whoami`" = "talukn" ]; then 
    PROJ_DIR=/mnt/d/home/talukn/graph_mining_cuda/parallel-sub-graph-mining-on-gpu
  fi
fi


GSPAN_CUDA="${PROJ_PATH}/src/gspan_cuda"

PARBIN=${PROJ_DIR}/bin
RESDIR=${PROJ_DIR}/results

LOGDIR=${PROJ_DIR}/logdir
CNFDIR=${PROJ_DIR}/cnf
SCRIPTDIR=${PROJ_DIR}/scripts
DATADIR=${PROJ_DIR}/testdata

#CUDA_EXECFILE=${PARBIN}/gspan_cuda
#CUDA_EXECFILE_NO_SORT=${PARBIN}/gspan_cuda_no_sort
#CUDA_EXECFILE_INTERSECTION=${PARBIN}/gspan_cuda_lists

SEQ_EXECFILE=${PARBIN}/gspan_seq


###################
# PARALLEL
#
###################

function remove_dat_suffix()
{
    local ARG1=$1;
    ARG1="$(basename ${ARG1} .cudat)"
    ARG1="$(basename ${ARG1} .dat)"
    echo -n ${ARG1}
}

function get_par_log_filename()
{
    local SUPPORT="$1"
    local INPUT_GRAPHDB="$(remove_dat_suffix $2)" 
    local RUNID="$3"


    if [ ! -d ${LOGDIR} ] ; then
    	mkdir -p ${LOGDIR}
    fi

    local LOGFILE="log_par-D${INPUT_GRAPHDB}-S${SUPPORT}-${RUNID}"
    echo -n ${LOGDIR}/${LOGFILE}
}

function get_par_targetname()
{
    local SUPPORT="$1"
    local INPUT_GRAPHDB="$(remove_dat_suffix $2)" 
    local RUNID=$3

    local TARGET="tgt-D${INPUT_GRAPHDB}-S${SUPPORT}-$3"
    echo -n ${TARGET}
}

function get_par_makefile_name()
{
    local INPUT_GRAPHDB="$(remove_dat_suffix $1)" 
    local RUNID="$2"

    local MAKEFILE="makefile_par-D${INPUT_GRAPHDB}-${RUNID}.mk"
    echo -n ${MAKEFILE}
}

function get_par_script_filename()
{
    local SUPPORT="$1"
    local INPUT_GRAPHDB="$(remove_dat_suffix $2)" 
    local RUNID=$3

    if [ ! -d ${SCRIPTDIR} ] ; then
	mkdir -p ${SCRIPTDIR}
    fi
    local SCRIPT="par-D${INPUT_GRAPHDB}-S${SUPPORT}-${RUNID}.sh"
    echo -n "${SCRIPTDIR}/${SCRIPT}"
}


#################
##  SEQUENTIAL 
##
#################

function get_seq_log_filename()
{
    local SUPPORT="$1"
    local INPUT_GRAPHDB="$(remove_dat_suffix $2)" 

    if [ ! -d ${LOGDIR} ] ; then
    	mkdir -p ${LOGDIR}
    fi

    local LOGFILE="log_seq-D${INPUT_GRAPHDB}-S${SUPPORT}"
#    echo ${LOGDIR}/${LOGFILE} 1>&2
    echo -n ${LOGDIR}/${LOGFILE}
}

function get_seq_targetname()
{
    local SUPPORT="$1"
    local INPUT_GRAPHDB="$(remove_dat_suffix $2)" 

    local TARGET="tgt-D${INPUT_GRAPHDB}-S${SUPPORT}"
    echo -n ${TARGET}
}

function get_seq_makefile_name()
{
    local INPUT_GRAPHDB="$(remove_dat_suffix $1)" 

    local MAKEFILE="makefile_seq-D${INPUT_GRAPHDB}.mk"
    echo -n ${MAKEFILE}
}

function get_seq_script_filename()
{
    local SUPPORT="$1"
    local INPUT_GRAPHDB="$(remove_dat_suffix $2)" 

#    echo get_par_script_filename $* 1>&2

    if [ ! -d ${SCRIPTDIR} ] ; then
	mkdir -p ${SCRIPTDIR}
    fi
    local SCRIPT="seq-D${INPUT_GRAPHDB}-S${SUPPORT}.sh"
    echo -n "${SCRIPTDIR}/${SCRIPT}"
}

