BEGIN {
    last = -1;
    found=0;
}

{
    if(found == 1) {
        print $1
        id = $1 + 0;
#        if(id != (last+1)) {
#            printf("gap: %d; last: %d\n", $1, last);
#        }
        last = $1;
        found = 0;
    }
}


$0 ~ /NSC/ {
    found = 1;
#    print "found";
}


