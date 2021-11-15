#!/bin/bash

MATRICES=`realpath $1`

OUTPUT_DIR=$2

set -e
test 2 -eq $# || { echo "Err in arg num"; exit 1; }  

cd $OUTPUT_DIR

{
read
while read line
do
link=$(echo $line | tr '|' ' ' | awk '{print $2}')
echo $link
wget $link
done
} < $MATRICES

for file in *.tar.gz
do
    tar -xzf "$file"
done

