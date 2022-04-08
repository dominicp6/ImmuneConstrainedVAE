#!/bin/bash

#path_to_muscle=$1
#file_prefix_name=$2
#start_number=$3
#end_number=$4
#
#counter=$start_number
#echo Aligning sequences
#while [ $counter -le $end_number ]
#do
#  echo Merging file number $counter
#  $path_to_muscle -profile -in1 ${file_prefix_name}_${start_number},${end_number}.fasta -in2 ${file_prefix_name}_${counter}.fasta -out ${file_prefix_name}_${start_number},${end_number}.fasta
#  ((counter++))
#done
#echo All done

/home/dominic/miniconda3/pkgs/muscle-3.8.1551-h7d875b9_6/bin/muscle -profile -in1 gen_and_natural_1_0.afa -in2 gen_and_natural_1_1.afa -out gan0,1.afa
/home/dominic/miniconda3/pkgs/muscle-3.8.1551-h7d875b9_6/bin/muscle -profile -in1 gen_and_natural_1_2.afa -in2 gen_and_natural_1_3.afa -out gan2,3.afa
/home/dominic/miniconda3/pkgs/muscle-3.8.1551-h7d875b9_6/bin/muscle -profile -in1 gen_and_natural_1_4.afa -in2 gen_and_natural_1_5.afa -out gan4,5.afa

/home/dominic/miniconda3/pkgs/muscle-3.8.1551-h7d875b9_6/bin/muscle -profile -in1 gan0,1.afa -in2 gan2,3.afa -out gan0,3.afa

/home/dominic/miniconda3/pkgs/muscle-3.8.1551-h7d875b9_6/bin/muscle -profile -in1 gan0,3.afa -in2 gan4,5.afa -out gan0,5.afa
