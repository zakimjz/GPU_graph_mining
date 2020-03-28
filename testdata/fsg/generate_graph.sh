#!/bin/bash


if [ $# -ne 6 ]
then
  echo "Usage: `basename $0` <#trans.>  <#freq patterns>  <#edge labels>  <#vertex labels>  <avg pattern size>  <avg trans. size>"
  exit $E_BADARGS
fi


NUM_TRANS=$1
NUM_PATTERNS=$2
EDGE_LABELS=$3
VERTEX_LABELS=$4
AVG_PATTERN_SIZE=$5
AVG_TRANS_SIZE=$6
OVERLAPPING="YES"

#./ggen -D10000 -L200 -E40 -V40 -i6 -T10 -S
if [ $OVERLAPPING = "YES" ]; then
 ./ggen -D${NUM_TRANS} -L${NUM_PATTERNS} -E${EDGE_LABELS} -V${VERTEX_LABELS} -i${AVG_PATTERN_SIZE} -T${AVG_TRANS_SIZE} -S > tmp.txt
else
 ./ggen -D${NUM_TRANS} -L${NUM_PATTERNS} -E${EDGE_LABELS} -V${VERTEX_LABELS} -i${AVG_PATTERN_SIZE} -T${AVG_TRANS_SIZE} > tmp.txt
fi


