#!/bin/bash


python /app/processor.py $1 $2 $3 $4 $5 $6 $7 $8 $9
ls /app/train_mask_png
ls -la /app/
python /app/upload_results.py $1 $2 $3 $4 $5 $6 $7 $8 $9
