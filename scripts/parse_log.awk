#!/bin/awk

BEGIN {
    FS="[\[]";
}

#/\[VAL\].*\[total_time=.*\]/{
#    split($7, a, /[=\]]/);
#    print a[2];
#  }

/\[VAL\].*\[[^=]*=.*]/ {
    split($7, S, /[=\]]/);
    if(S[1]==var_name) {
	print S[2];
    } # if
} 

