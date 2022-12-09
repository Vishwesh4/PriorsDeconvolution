#!/bin/bash

#run all the experiments for generating table in the report
nohup python get_metrics.py -b 0 -l 0 -n 0 --gpu_id 0 > ./results/exp1_1.out &
nohup python get_metrics.py -b 1 -l 0 -n 0 --gpu_id 0 > ./results/exp2_1.out &
nohup python get_metrics.py -b 2 -l 0 -n 0 --gpu_id 0 > ./results/exp3_1.out &
nohup python get_metrics.py -b 0 -l 1 -n 0 --gpu_id 0 > ./results/exp4_1.out &
nohup python get_metrics.py -b 1 -l 1 -n 0 --gpu_id 0 > ./results/exp5_1.out &
nohup python get_metrics.py -b 2 -l 1 -n 0 --gpu_id 0 > ./results/exp6_1.out &
nohup python get_metrics.py -b 0 -l 0 -n 1 --gpu_id 0 > ./results/exp7_1.out &
nohup python get_metrics.py -b 1 -l 0 -n 1 --gpu_id 0 > ./results/exp8_1.out &
nohup python get_metrics.py -b 2 -l 0 -n 1 --gpu_id 2 > ./results/exp9_1.out &
nohup python get_metrics.py -b 0 -l 1 -n 1 --gpu_id 2 > ./results/exp10_1.out &
nohup python get_metrics.py -b 1 -l 1 -n 1 --gpu_id 2 > ./results/exp11_1.out &
nohup python get_metrics.py -b 2 -l 1 -n 1 --gpu_id 2 > ./results/exp12_1.out &
