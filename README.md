# Undergraduate Thesis
This repository contains my undergraduate thesis, [**Supervised Monocular Depth Estimation via Stacked Generalization**](https://drive.google.com/file/d/1wrjHJrAY8h8B7ccvu8Z7Y0r3zFU-cVUI/view?usp=sharing).

## Update
- [28 Dec 2021]: [**One page summary and presentation slide with new updates**](https://github.com/sincostanx/yylab/tree/main/yy-lab-2021-%E5%90%88%E5%90%8C%E3%82%BC%E3%83%9F%20(%E6%9D%B1%E5%B7%A5%E5%A4%A7%2C%20%E8%BE%B2%E5%B7%A5%E5%A4%A7%2C%20%E9%9B%BB%E9%80%9A%E5%A4%A7%2C%20%E5%90%8D%E5%B7%A5%E5%A4%A7)) Implementations and detailed discussions cannot be published due to university's regulation.

## Note
- ***This is a work in progress.*** All experimental data and implementations have not been finalized.
- Due to technical issues during the experiment, parts of the implementation cannot be recovered. Therefore, the experiment **SG: Simultaneously** and **SG: RGB, Tuned** cannot be replicated (Please refer to the [report](https://drive.google.com/file/d/1wrjHJrAY8h8B7ccvu8Z7Y0r3zFU-cVUI/view?usp=sharing) for definition).

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
- Execute the following command to train <model_name>
```
python3 train.py args_train_nyu_<model_name>.txt
```
Valid <model_name> are **adabins**, **bts**, **ldrn**, and **controller** (the meta-learner). Note that controller requires 3 pretrained base learners (./checkpoints/\*_best.pt). Please change the path in the code manually (for now).
- To replicate experiment **SG: RGB, Freeze**, **SG: D**, and **SG: RGB + D**, use argument controller-input as **i**, **o**, and **io**, respectively.
- When changing hyperparameters, please use **lr-finder.py** to find the appropriate learning rate.
```
python3 lr-finder.py args_train_nyu_<model_name>.txt
```

### 2. Evaluating
- Execute the following command to train <model_name>
```
python3 evaluate.py args_eval_nyu_<model_name>.txt
```
Valid <model_name> are **adabins**, **bts**, **ldrn**, **controller**, and **baseline** (unweighted average).
- Change argument **checkpoint_path** accordingly when evaluating a newly trained model. The default values are pretrained models provided in the link above.

### 3. Inspect Network Architecture
You can inspect network architectures by using the command
```
python3 summary.py --inspect-module <module> --controller-input <input>
```
Valid <module> are **adabins**, **bts**, **ldrn**, **controller**, and **ensemble**. Valid <input> are **i**, **o**, and **io**. You can ignore argument **--controller-input** when inspect base learners.
 
## Future works
- Quantitative analysis (inference)
- Experiment with
    - Base learners without significant performance gap
    - Larger encoders with more representation capability
    - Different evaluation (average between original and flipped image)
