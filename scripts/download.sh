#!/bin/bash

MATRICES=$1

test 1 -eq $# || { echo "Err in arg num"; exit 1; }  

{
read
while read line
do
link=$(echo $line | tr '|' ' ' | awk '{print $5}')
python3 ./download.py -i $link
done
} < $MATRICES

cd /home/hajta2/Downloads
for file in *.tar.gz
do
    tar -xzf "$file" -C /home/hajta2/Desktop/Egyetem/PGA/Matrices
done