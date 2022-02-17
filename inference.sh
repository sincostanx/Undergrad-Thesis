#!/bin/bash

<<COMMENT
'experimental_setup/inference/meta-learner/CV-adabins_IO-F-D161.txt'
COMMENT

setups=(
'experimental_setup/inference/meta-learner/MOD_baseline.txt'
'experimental_setup/inference/meta-learner/MOD_I-F-G.txt'
'experimental_setup/inference/meta-learner/MOD_O-F-G.txt'
'experimental_setup/inference/meta-learner/MOD_IO-F-G.txt'
'experimental_setup/inference/meta-learner/MOD_IO-F-M2.txt'
'experimental_setup/inference/meta-learner/MOD_IO-F-D161.txt'
'experimental_setup/inference/meta-learner/SAME_baseline.txt'
'experimental_setup/inference/meta-learner/SAME_IO-F-G.txt'
'experimental_setup/inference/meta-learner/SAME_IO-F-D161.txt'
'experimental_setup/inference/meta-learner/CV-BTS_baseline.txt'
'experimental_setup/inference/meta-learner/CV-BTS_IO-F-G.txt'
'experimental_setup/inference/meta-learner/CV-BTS_IO-F-D161.txt'
'experimental_setup/inference/meta-learner/CV-LDRN_baseline.txt'
'experimental_setup/inference/meta-learner/CV-LDRN_IO-F-G.txt'
'experimental_setup/inference/meta-learner/CV-LDRN_IO-F-D161.txt'
'experimental_setup/inference/meta-learner/CV-adabins_baseline.txt'
'experimental_setup/inference/meta-learner/CV-adabins_IO-F-G.txt'
'experimental_setup/inference/baselearner/MOD-adabins.txt'
'experimental_setup/inference/baselearner/MOD-BTS.txt'
'experimental_setup/inference/baselearner/MOD-LDRN.txt'
'experimental_setup/inference/baselearner/SAME-1.txt'
'experimental_setup/inference/baselearner/SAME-2.txt'
'experimental_setup/inference/baselearner/SAME-3.txt'
'experimental_setup/inference/baselearner/SAME-4.txt'
'experimental_setup/inference/baselearner/SAME-5.txt'
'experimental_setup/inference/baselearner/CV-adabins-1.txt'
'experimental_setup/inference/baselearner/CV-adabins-2.txt'
'experimental_setup/inference/baselearner/CV-adabins-3.txt'
'experimental_setup/inference/baselearner/CV-adabins-4.txt'
'experimental_setup/inference/baselearner/CV-adabins-5.txt'
'experimental_setup/inference/baselearner/CV-BTS-1.txt'
'experimental_setup/inference/baselearner/CV-BTS-2.txt'
'experimental_setup/inference/baselearner/CV-BTS-3.txt'
'experimental_setup/inference/baselearner/CV-BTS-4.txt'
'experimental_setup/inference/baselearner/CV-BTS-5.txt'
'experimental_setup/inference/baselearner/CV-LDRN-1.txt'
'experimental_setup/inference/baselearner/CV-LDRN-2.txt'
'experimental_setup/inference/baselearner/CV-LDRN-3.txt'
'experimental_setup/inference/baselearner/CV-LDRN-4.txt'
'experimental_setup/inference/baselearner/CV-LDRN-5.txt'
)

for setup in "${setups[@]}"; do
    python3 -W ignore infer.py "$setup"
done
