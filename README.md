## [CVPR 2022] No Pain, Big Gain: Classify Dynamic Point Cloud Sequences With Static Models by Fitting Feature-Level Space-Time Surfaces

## License
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

## Citation
If you find this work useful in your research, please cite:
```
@InProceedings{Zhong_2022_CVPR,
    author    = {Zhong, Jia-Xing and Zhou, Kaichen and Hu, Qingyong and Wang, Bing and Trigoni, Niki and Markham, Andrew},
    title     = {No Pain, Big Gain: Classify Dynamic Point Cloud Sequences With Static Models by Fitting Feature-Level Space-Time Surfaces},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8510-8520}
}
```

## Abstract
Scene flow is a powerful tool for capturing the motion field of 3D point clouds. However, it is difficult to directly apply flow-based models to dynamic point cloud classification since the unstructured points make it hard or even impossible to efficiently and effectively trace point-wise correspondences. To capture 3D motions without explicitly tracking correspondences, we propose a kinematics-inspired neural network (Kinet) by generalizing the kinematic concept of ST-surfaces to the feature space. By unrolling the normal solver of ST-surfaces in the feature space, Kinet implicitly encodes feature-level dynamics and gains advantages from the use of mature backbones for static point cloud processing. With only minor changes in network structures and low computing overhead, it is painless to jointly train and deploy our framework with a given static model. Experiments on NvGesture, SHREC'17, MSRAction-3D, and NTU-RGBD demonstrate its efficacy in performance, efficiency in both the number of parameters and computational complexity, as well as its versatility to various static backbones.

## Installation
Install <a href="https://www.tensorflow.org/install/pip">TensorFlow 1.x-GPU</a> for training models & <a href="https://pytorch.org/get-started/previous-versions/">Pytorch-CPU</a> for reading data. The code is runnable under TensorFlow 1.9.0 GPU version, Pytorch 1.1.0 CPU version, and Python 3.6. It's highly recommended that you have access to GPUs.

### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them first by `make` under each ops subfolder (check `Makefile`) or directly use `sh command_make.sh`. **Update** `arch` **in the Makefiles for different** <a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported">CUDA Compute Capability</a> **that suits your GPU if necessary**.

## Gesture Recognition on SHREC'17 without Bounding Boxes
### Data Preparation
We provide the point cloud data of ***SHREC'17 without Bounding Boxes*** [here](https://github.com/jx-zhong-for-academic-purpose/Kinet/releases/download/v1.0/BackgroundSHREC2017.7z), which (`BackgroundSHREC2017`) should be downloaded to `./dataset/SHREC2017`. Following previous researches on the same dataset with bounding boxes, we generate 256 points per frame and sample 128 for training a 32-frame classifier. The data splitting files can be found as follows - download the [SHREC'17 dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/) and put `HandGestureDataset_SHREC2017` directory to `./dataset/SHREC2017`, It is suggested to make a soft link toward the downloaded dataset.
### Train
First, let's train the static branch (For better performance, pretraining on additional static datasets like *ModelNet40* is probably helpful):
```
cd shrec2017_release
bash train_static.sh
```
Then, train the dynamic branch:
```
bash train_dynamic.sh
```
### Test a Trained Model
A trained model is provided [here](https://github.com/jx-zhong-for-academic-purpose/Kinet/releases/download/v1.0/trained_dynamic_model.zip). The performance of static, dynamic & fusion branches can be evaluated:
```
bash test.sh
```

## Data Preparation & Experiments on Other Datasets
You can easily adapt our code by using the datasets in the following related projects & their dataloaders.

## Related Projects
* <a href="https://arxiv.org/abs/1910.09165" target="_blank">MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences
</a> by Liu et al. (ICCV 2019). The dataloader for ***MSRAction-3D*** is available <a href="https://github.com/xingyul/meteornet">here</a>.
* <a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Min_An_Efficient_PointLSTM_for_Point_Clouds_Based_Gesture_Recognition_CVPR_2020_paper.html" target="_blank">An Efficient PointLSTM for Point Clouds Based Gesture Recognition
</a> by Min et al. (CVPR 2020). The dataloaders for ***NvGesture*** & ***SHREC'17 (with Bounding Boxes)*** are available <a href="https://github.com/ycmin95/pointlstm-gesture-recognition-pytorch">here</a>.
* <a href="https://openreview.net/pdf?id=O3bqkf_Puys" target="_blank">PSTNet: Point Spatio-Temporal Convolution on Point Cloud Sequences</a> by Fan et al. (ICLR 2021). The dataloader for ***NTU-RGBD*** is available <a href="https://github.com/hehefan/Point-Spatio-Temporal-Convolution">here</a>.
