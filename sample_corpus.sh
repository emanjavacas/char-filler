#!/bin/bash

# input is a dir with files with sentences
INPUT=$1
shift

# rest args are split proportions adding up to 100
NUMARGS=$#
if [ $NUMARGS -eq 0 ]; then
   echo "missing proportions";
   exit;
fi

PROPS=0
i=1
for arg
do
    if [ $(($i%2)) -eq 0 ]; then
	PROPS=$(($PROPS + arg));
    fi
    let i++;
done
 
if [ $PROPS -ne 100 ]; then
    echo "proportions must add to 100; sum: $PROPS";
    exit;
fi

# total num of lines
LINES=$(cat $INPUT/* | wc -l)

# shuffle to a tmp file
TMP=$(mktemp)
cat $INPUT/* | shuf > $TMP

# partition over files
{
    i=1
    for arg
    do
	if [ $(($i%2)) -eq 0 ]; then
	    OUTNAMEARG=$((i-1))
	    OUTNAME=${!OUTNAMEARG};
	    PROP=${!i}
	    echo "Piping $((($LINES*$PROP)/100)) lines to file $OUTNAME.txt";
	    head -n $((($LINES*$PROP)/100)) > $OUTNAME.txt
	fi
	let i++
    done
} < $TMP

# clean up
rm $TMP
