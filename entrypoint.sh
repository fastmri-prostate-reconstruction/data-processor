#!/bin/bash


python /app/processor.py $1 $2 $3 $4 $5 $6 $7 $8 $9
ls -la ~/data
ls -la ~/data/train_mask_png
python /app/upload_results.py $1 $2 $3 $4 $5 $6 $7 $8 $9
