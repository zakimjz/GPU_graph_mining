#! /bin/bash

. ./local_vars.sh


function get_seq_makefile_target_name()
{
  local SUPPORT="$1"
  local INPUT_GRAPHDB="$2"
  TARGET=` get_seq_targetname ${SUPPORT} ${INPUT_GRAPHDB} `
  echo -n ${TARGET}
}


function get_seq_makefile_target_content()
{
  local SUPPORT="$1"
  local INPUT_GRAPHDB="$2"
  local FILETYPE="$3"
  LOGFILE="` get_seq_log_filename ${SUPPORT} ${INPUT_GRAPHDB}`"

  echo
  echo "`get_seq_makefile_target_name ${SUPPORT} ${INPUT_GRAPHDB}`:"
  #echo -e "\t" ${SEQ_EXECFILE} "${SUPPORT} ${INPUT_GRAPHDB} > ${LOGFILE}"
  echo -e "\t${SEQ_EXECFILE} ${FILETYPE} ${INPUT_GRAPHDB} ${SUPPORT} > ${LOGFILE}"
  echo
}



function get_seq_makefile_targets()
{
  local ARGS="$@"
  local INPUT_GRAPHDB="$1"
   # shift first  parameter, the remaining one is the support list
  shift; 
  local DEPS=""

  for support in "$@"
  do
    DEPS="${DEPS} `get_seq_makefile_target_name $support  ${DATADIR}/${INPUT_GRAPHDB}`"
  done

  echo "all: ${DEPS}"
  echo
}

function get_seq_makefile_content()
{
  local ARGS="$@"
  local INPUT_GRAPHDB="$1"
  local FILETYPE="$2"
   # shift first 2 parameters, the remaining one is the support list
  shift; shift;

  for support in "$@"
  do
    get_seq_makefile_target_content $support ${DATADIR}/${INPUT_GRAPHDB} ${FILETYPE}
  done

}


#get_makefile_targets -protein ${DATADIR}/uniprot_sprot.seq.200k 190000  195000 195000

#CONTENT =`get_makefile_content`
#echo "all: ${DEPS}" > ${SCRIPTDIR}/exec_seq_experiments.mk


