# Undergraduate Thesis
This repository contains my undergraduate thesis, [**Supervised Monocular Depth Estimation via Ensemble Deep Learning**](https://drive.google.com/file/d/1V59fh9bmv2NEmJhtu929xitrKYqfmdPu/view?usp=sharing).
Presentation slide is available [here](https://docs.google.com/presentation/d/1l2YEqjiCr4zkmqZ9Y_KuauE-mtg-l4Bc4Hxx6Hm1RjI/edit?usp=sharing). To comply with the regulations of the university, please contact me if you would like to use pretrained models.

## Note
- **This implementation is refactored from the one used during the experiment (for the sake of readability and such).** Although I have tested it several times to ensure that there is no bug, please contact me if it shows unexpected behaviors.
- Due to technical issues during the experiment, parts of the implementation cannot be recovered. Therefore, the experiment **MOD_S** and **MOD_I-T-G** cannot be replicated (Please refer to the report above for definition).

## Environment
This project uses the following libraries and frameworks. To avoid any dependency issues, please set up the environment accordingly.
- **Pytorch** 1.7.1+cu10.1 with **Torchvision** 0.8.2
- **Torchinfo** 1.5.2
- **Anaconda** 4.10.3
- **Wandb** 0.10.33
- **Pytorch3D** 0.4.0
- **Numpy** 1.19.2
- **Python** 3.6.9

## NYU V2 Dataset
- Training data can be downloaded from [here](https://drive.google.com/uc?export=download&id=1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP) (Credit to [bts](https://github.com/cogaplex-bts/bts/tree/master/pytorch)).
- Testing data for evaluation can be downloaded from [here](https://drive.google.com/file/d/1JJoUTB94pCqGuHRtDZkBY2KFavEcThfZ/view?usp=sharing).

## Overview
- Clone this repository
- Download training and testing data. In **args_\*.txt**, change argument **data_path** and **gt_path** to training data path and **data_path_eval** and **gt_path_eval** to testing data path
- Pretrained base learners and controllers from experiment **SG: RGB, Freeze**, **SG: D**, and **SG: RGB + D** can be downloaded from [here](https://drive.google.com/file/d/1ZUu8X38EO9uS2pWEvVJ4X0gkqRAvDaLx/view?usp=sharing). Create directory **checkpoints** and put those files in it.

### 1. Training
- Execute the following command to train <experiment_name>
```
python3 train.py ./experimental_setup/train/<category>/<experiment_name>.txt
```
Valid <category> are **baselearner** and **meta-learner**. Please refer to the report for the name of each experiment.
- When changing hyperparameters, please use **lr-finder.py** to find the appropriate learning rate.

### 2. Evaluating
- Execute the following command to evaluate <experiment_name>
```
python3 evaluate.py ./experimental_setup/evaluate/<category>/<experiment_name>.txt
```
- Change argument **checkpoint_path** accordingly when evaluating a newly trained model.

### 3. Inferring
- Execute the following command to infer using <experiment_name>
```
python3 infer.py ./experimental_setup/infer/<category>/<experiment_name>.txt
```
or alternartively, execute the following bash scrpt to infer using all <experiment_name> available in this work.
```
./inference.sh
```

### 4. Inspect Network Architecture
You can inspect network architectures by using the command
```
python3 summary.py ./experimental_setup/evaluate/<category>/<experiment_name>.txt
```
or alternartively, execute the following bash scrpt to describe every different network architectures used in this work.
```
./architecture.sh
``` 
