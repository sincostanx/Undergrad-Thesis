#!/bin/bash

setups=(
'experimental_setup/inference/meta-learner/MOD_I-F-G.txt'
'experimental_setup/inference/meta-learner/MOD_O-F-G.txt'
'experimental_setup/inference/meta-learner/MOD_IO-F-G.txt'
'experimental_setup/inference/meta-learner/MOD_IO-F-M2.txt'
'experimental_setup/inference/meta-learner/MOD_IO-F-D161.txt'
'experimental_setup/inference/meta-learner/SAME_IO-F-G.txt'
'experimental_setup/inference/meta-learner/SAME_IO-F-D161.txt'
'experimental_setup/inference/baselearner/MOD-adabins.txt'
'experimental_setup/inference/baselearner/MOD-BTS.txt'
'experimental_setup/inference/baselearner/MOD-LDRN.txt'
)

for setup in "${setups[@]}"; do
    python3 -W ignore summary.py "$setup"
done
