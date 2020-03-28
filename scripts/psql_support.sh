#!/bin/bash




function psql_create_experiment()
{
    local FILENAME=$1
    local RUNID=$2

    local EXPCOUNT=$(psql -P tuples_only -P format=unaligned -c "select count(idexp) from experiment_description where gspan_variant='${RUNID}' and date='${DB_TIMESTAMP}';")

    if [ ${EXPCOUNT} -ge 1 ] ; then
	echo "error: experiment already created";
	return;
    fi

    psql -P tuples_only -c "insert into experiment_description (description,gspan_variant,date) values('${DESCRIPTION}', '${RUNID}', '${DB_TIMESTAMP}');" 1>/dev/null
}

function psql_insert_database()
{
    local DBFILE=$1

    local DBFILEEXISTS=$(psql -P tuples_only -c "select count(iddb) from databases where name='${DBFILE}'")
    if [ ${DBFILEEXISTS} -ge 1 ] ; then
	echo "error: database file already exists";
	return;
    fi
    psql -P tuples_only -c "insert into databases (name) values('${DBFILE}');"
}

function psql_get_database_id()
{
    local DBFILE=$1
    local DBFILEID=$(psql -P tuples_only -P format=unaligned -c "select iddb from databases where name='${DBFILE}'")
    echo ${DBFILEID}
}


function psql_get_experiment_id()
{
    local RUNID=$1
    local EXPID=$(psql -P tuples_only -P format=unaligned -c "select idexp from experiment_description where date='${DB_TIMESTAMP}' and gspan_variant='${RUNID}'")
    echo ${EXPID}
}

function psql_experiment_exists()
{
    local FILENAME=$1
    local RUNID=$2
    local SUPPORT=$3

    local DBID=$(psql_get_database_id ${FILENAME})
    local EXPID=$(psql_get_experiment_id ${RUNID})
    if [ "x${DBID}" = "x" -o "x${EXPID}" = "x" ] ; then
	echo "";
	return;
    fi

#    echo "select count(idexp) from experiment_result where iddb=${DBID} and idexp='${EXPID}' and support='${SUPPORT}'"

    local COUNT=$(psql -P tuples_only -P format=unaligned -c "select count(idexp) from experiment_result where iddb=${DBID} and idexp='${EXPID}' and support='${SUPPORT}'")
    echo ${COUNT}
}

function psql_process_experiment()
{
    local FILENAME=$1
    local RUNID=$2
    local support=$3
    local SEQ_TOTAL_TIME=$4
    local PAR_TOTAL_TIME=$5

    local IDDB=$(psql_get_database_id ${FILENAME})
    local EXPID=$(psql_get_experiment_id ${RUNID})
    local SEQ_EXPID=$(psql_get_experiment_id seq)
#    echo "${SEQ_EXPID}"
    if [ "x${SEQ_EXPID}" = "x" ] ; then
	echo "experiment seq does not exists"
	psql_create_experiment ${FILENAME} seq
    fi

    local SEQ_EXPRESULT=$(psql_experiment_exists ${FILENAME} seq ${support})
#    echo "SEQ_EXPRESULT: ${SEQ_EXPRESULT}"
    if [ ${SEQ_EXPRESULT} -eq 0 ] ; then
	SEQ_EXPID=$(psql_get_experiment_id seq)
	echo "creating sequential experiment (${IDDB}, ${SEQ_EXPID}, ${support}, ${SEQ_TOTAL_TIME})"
	psql -P tuples_only -P format=unaligned -c "insert into experiment_result values (${IDDB}, ${SEQ_EXPID}, ${support}, ${SEQ_TOTAL_TIME});"
    fi

    local PAR_EXPRESULT=$(psql_experiment_exists ${FILENAME} ${RUNID} ${support})
    if [ ${PAR_EXPRESULT} -eq 0 ] ; then
	echo "creating parallel experiment (${IDDB}, ${EXPID}, ${support}, ${PAR_TOTAL_TIME})"
	psql -P tuples_only -P format=unaligned -c "insert into experiment_result values (${IDDB}, ${EXPID}, ${support}, ${PAR_TOTAL_TIME});"
    fi
}
