#!/bin/bash

for i in $(seq $2 $3);
do python3 ./ddgun_seq.py ./$1/${i}_reference.fasta ./$1/$i.muts > ./$1/$i.txt;
done
