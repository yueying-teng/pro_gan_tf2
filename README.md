## TensorFlow 2 Implementation of Progressive Growing of GANs
### 📄[Paper](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf)


### How to use
#### 1. Build and run the container
```bash
bash bin/docker_build.sh

bash bin/docker_run.sh
```

#### 2. Prepare the data
- For this project, the training set (both male and female) of CELEBA HQ data is used, and it's downloaded from [Kaggle](https://www.kaggle.com/lamsimon/celebahq).
- Put the downloaded data at the root dir of the project

#### 3. Train the models
```bash
bash bin/log_into_container.sh
cd notebooks

# run the training script; logs will be exported to anohter file
python3 train_pro_gan.py --data_dir "/work/CELEBAHQ/train" --save_dir "./test_train" --epochs 40 > pro_gan_training.log 2>&1
```

#### 4. Start tensorboard
```bash
# open a new terminal window
bash bin/log_into_container.sh

# NOTE that the log dir below should be replaced with the save_dir/logs you specified
bash bin/start_tensorboard.sh /work/notebooks/test_train/logs
```

### Project structure
```
- pro_gan_tf
    - bin
    - CELEBAHQ  <-- replace this with your data
    - config
    - demo
    – docker
    - notebooks   <-- train_pro_gan.py is here, logs are also saved here while training
        - train_pro_gan.py
        - test_train
            - logs/20220105-142028
            - models/20220105-142028
    - pro_gan
    - README.md
    - requirements.txt
    - test
    - training_results
```

### To create demo videos
```bash
# open a new terminal window
bash bin/log_into_container.sh
cd demo

# create a video using the images saved during training
python3 create_video_from_training_feedback.py --log_dir "/work/notebooks/test_train/logs/20220105-142028" --depth 7

# create a video using the images generated from the interpolated latent space vectors
python3 latent_space_interpolation.py --model_dir "/work/notebooks/test_train/models/20220105-142028" --video_length 10 --depth 7 --num_videos 4

# use the trained generator model to create a sheet of generated images
python3 create_facesheet.py --model_dir "/work/notebooks/test_train/models/20220105-142028" --num_samples 16 --depth 7 --num_sheets 4
```

### Some training results
<img src="https://github.com/yueying-teng/pro_gan_tf/blob/master/training_results/gen_loss.png" width="820" >

<img src="https://github.com/yueying-teng/pro_gan_tf/blob/master/training_results/dis_loss.png" width="820" >

- One Tesla P40 is used for training the models up till resolution 128 x 128 before running out of memory.
- This training process took about 25 days and smaller batch sizes are used for training models with image resolution 64 and above.

**Latent space interpolation**

<img src="https://github.com/yueying-teng/pro_gan_tf/blob/master/training_results/000_latent_space_interpolation_resolution_128.gif" width="200" height="200" /> <img src="https://github.com/yueying-teng/pro_gan_tf/blob/master/training_results/001_latent_space_interpolation_resolution_128.gif" width="200" height="200" /> <img src="https://github.com/yueying-teng/pro_gan_tf/blob/master/training_results/002_latent_space_interpolation_resolution_128.gif" width="200" height="200" /> <img src="https://github.com/yueying-teng/pro_gan_tf/blob/master/training_results/003_latent_space_interpolation_resolution_128.gif" width="200" height="200" />


**Facesheet 128 x 128**

<p align="center">
  <img src="https://github.com/yueying-teng/pro_gan_tf/blob/master/training_results/000_facesheets_resolution_128.png" width="420">
&nbsp; &nbsp;
  <img src="https://github.com/yueying-teng/pro_gan_tf/blob/master/training_results/001_facesheets_resolution_128.png" width="420">
</p>

### Generator model w/ resolution 128 x 128
Click [here](https://drive.google.com/drive/folders/1y0x-Sf5pFM1BqWhLKAuDb5NdOmqoANAX?usp=sharing) to download the model.

To generate facesheets and latent space interpolation videos, put the downloaded model at `notebooks/test_train/models/`.

### References:
- [akanimax/pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch/tree/v3.0)
- [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py)
- [tensorflow_gan/python/losses/losses_impl](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py#L395)
- [tensorflow_gan/examples/progressive_gan](https://github.com/tensorflow/gan/blob/696f06c49fd598fa3397039a28e597b0b26c43ed/tensorflow_gan/examples/progressive_gan/layers.py#L184)


### TODOs:
- add support for
    - conditional GAN
    - continue training from any depth
    - EMA update of generator weights
- add metric FID
