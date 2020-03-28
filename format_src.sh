#!/bin/bash

#echo $0

CFGFILE=`dirname $0`/format_src.cfg

#astyle -Jpo -A4 -U -s2 $*
#uncrustify -c 

uncrustify -c $CFGFILE -l CPP --replace $*


