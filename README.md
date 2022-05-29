# Blockwise Spectral Analysis for Deepfake Detection in High-fidelity Videos

## Introduction
Most deepfake detection methods only target specific GANs and do not generalize well to other unseen GAN architectures, even though there are some current studies to detect deepfakes through spectral artifacts left by upsampling operations. In particular, they all fail on high-fidelity datasets. 

To address this, we crop fake images into informative patches and use their spectrum to train a classifier. Experimental results show that our method is effective in detecting high-fidelity deepfakes and generalizes well in unseen GANs.



## Installation 

Make a full clone to make sure cloning all the submodules.

```bash
git clone --recursive https://github.com/HollyhuangD/blockwise_spectral_analysis.git
```


## Set-up
### environment requirements:
My current conda environment is attached as [Blockwise.yml](conda/Blockwise.yml)
python >= 3.6
torch >= 1.1.0
```
conda env create -f ./conda/Blockwise.yml
```

### You can choose to download VFHQ dataset and extract 
VFHQ can be find on https://people.mpi-inf.mpg.de/~gfox/. Copy VFHQ dataset into folder ./extract/input/.

Extract datasets from VFHQ
```bash
# From project directory
# Clean ./extract/output DIR before each command

# Extract 'SMALL' dataset from VFHQ
python ./code/extract_frame.py --mode='SMALL'

# Extract 'LARGE' dataset from VFHQ
python ./code/extract_frame.py --mode='LARGE'

# Extract 'EM' dataset from VFHQ
python ./code/extract_frame.py --mode='EM'

# Extract 'FACE' dataset from VFHQ
python ./code/extract_frame.py --mode='FACE'
```

### Or you can download extracted dataset directly
Real dataset are provided ([Google Drive](https://drive.google.com/file/d/1iSKzIXNZUAJ1OThrF9Od5_HSUg9lB-Kc/view?usp=sharing)). Download and unzip to `./data/real` folder.
Fake dataset are provided ([Google Drive](https://drive.google.com/file/d/1MQ4W7quEmacLwZMfpIG-mVi6Se4BGh1f/view?usp=sharing)). Download and unzip to `./data/fake` folder.

### You can download pre-trained models
Pre-trained models are provided ([Google Drive](https://drive.google.com/file/d/1pkf2IYyO-ZnZbzRJAlJDiNauJRfjK5qd/view?usp=sharing)). Download and unzip to `./model_resnet` folder.


## How to train the model reported in the paper
```bash
# From project directory

# Train
python ./code/run_training.py --feature fft --gpu-id 0 --training_sets D_H D_S D_L D_EM D_F --test_sets horse zebra summer winter apple orange facades monet photo CycleGAN D_S D_L D_EM D_F
```

## How to test single model
### Run test
```bash
# From project directory
python ./code/run_test.py --feature fft --gpu-id 0 --training_sets D_H D_S D_L D_EM D_F --test_sets horse zebra summer winter apple orange facades monet photo CycleGAN D_L D_S D_EM D_F
```

