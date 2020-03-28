#! /bin/bash

. ./local_vars.sh


function get_par_makefile_target_name()
{
  local SUPPORT="$1"
  local INPUT_GRAPHDB="$2"
  local RUNID="$3"

  TARGET=` get_par_targetname ${SUPPORT} ${INPUT_GRAPHDB} ${RUNID} `
  echo -n ${TARGET}
}


function get_par_makefile_target_content()
{
  local CUDA_EXECFILE="$1"
  local SUPPORT="$2"
  local INPUT_GRAPHDB="$3"
  local RUNID="$4"
  local FILETYPE="$5"
  LOGFILE="` get_par_log_filename ${SUPPORT} ${INPUT_GRAPHDB} ${RUNID} `"

  echo
  echo "`get_par_makefile_target_name ${SUPPORT} ${INPUT_GRAPHDB} ${RUNID}`:"
  echo -e "\t${PARBIN}/${CUDA_EXECFILE} ${FILETYPE} ${INPUT_GRAPHDB} ${SUPPORT}  > ${LOGFILE}"
  echo
}



function get_par_makefile_targets()
{
  local ARGS="$@"
  local INPUT_GRAPHDB="$1"
  local RUNID="$2"

   # shift first 4 parameters, the remaining one is the support list
  shift 2;
  local DEPS=""

  for support in "$@"
  do
    DEPS="${DEPS} `get_par_makefile_target_name  $support  ${DATADIR}/${INPUT_GRAPHDB}  ${RUNID}`"
  done

  echo "all: ${DEPS}"
  echo
}

function get_par_makefile_content()
{
  local ARGS="$@"
  local CUDA_EXECFILE="$1"
  local INPUT_GRAPHDB="$2"
  local RUNID="$3"
  local FILETYPE="$4"

   # shift first 5 parameters, the remaining one is the support list
  shift 4;

  for support in "$@"
  do
    get_par_makefile_target_content ${CUDA_EXECFILE} $support ${DATADIR}/${INPUT_GRAPHDB} ${RUNID} ${FILETYPE}
  done

}


#get_makefile_targets -protein ${DATADIR}/uniprot_sprot.seq.200k 190000  195000 195000

#CONTENT =`get_makefile_content`
#echo "all: ${DEPS}" > ${SCRIPTDIR}/exec_seq_experiments.mk

#SUPPORT_LIST="10 20 30"
#INPUT_GRAPHDB="Chemical_340"
#CONFIG_FILE="dummy_config"
#CONFIG_ID="dummy_id"
#FILETYPE="txt"

#MAKEFILE="`get_par_makefile_name ${INPUT_GRAPHDB} ${CONFIG_FILE} ${CONFIG_ID}`"
#get_par_makefile_targets ${INPUT_GRAPHDB} ${CONFIG_FILE} ${CONFIG_ID}  ${SUPPORT_LIST} > ${SCRIPTDIR}/${MAKEFILE}
#get_par_makefile_content ${INPUT_GRAPHDB} ${CONFIG_FILE} ${CONFIG_ID} ${FILETYPE}  ${SUPPORT_LIST} >> ${SCRIPTDIR}/${MAKEFILE}

