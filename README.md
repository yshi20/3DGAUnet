## Introuction

* This is a pytorch implementation of part of 3DGAUnet 

### Prerequisites

* Python >= 3.7.9 
* Pytorch >= 1.6.0
* tensorboardX >=  2.1
* matplotlib >= 2.1


### Pipeline
#### Data

Volumetric data in .mat format should be placed in'/src/volumetric_data/', and change the directory in 'paramt.py' accordingly.

#### Training
`cd src`
run `python main.py` on GPU or CPU. Of course, you need a GPU for training until getting good results. I used one GeForce RTX 4090 in my experiments on 3D models with resolution of 64x64x64

model weights and some 3D reconstruction images would be logged to the `outputs` folders, respectively, for every `model_save_step` number of step in `paramt.py`. You can play with all parameters in `paramt.py`.

#### Generation of synthesis data
To generate volumetric data from trained model, you can run `python main.py --test=True` to call `tester.py`.

#### Pretrained Model
Pretrained models are in the `outputs` folder. Then run `python main.py --test=True --model_name=pancreas_pretrained`. You will find the outputs in the `test_outputs` folder within `pancreas_pretrained`.
