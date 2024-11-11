#!/bin/bash
dir=$(pwd)

nm="poly6"
cmd="poly6"

cd "$dir/dataset/$nm"  
python ../../imgGenerator.py 0 300 $cmd
rm -rf output

i=1
for file in *.png; do
    echo "Renaming ${nm}_${i}.png" 
    mv "$file" "${nm}_${i}.png"
    ((i++))
done
