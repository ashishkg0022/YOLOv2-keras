### YOLOv2-keras

This is a keras implementation of YOLOv2 (YOLO9000). Original Paper : [YOLO9000](https://arxiv.org/abs/1612.08242)

#### Requirements

Keras
TensorFlow
scipy
h5py
matplotlib

#### Test

To run the model download the weights from this [link](https://drive.google.com/drive/folders/1WjjuImQB0WbweNsbMcaOWSdqVFCKayS3) . And move this `weights.h5` inside `weights` folder. Weights from the original YOLO has been used.

Put your test image insied images folder and run the model.

(Some of the functions of `utils.py` has been used from [YAD2K](https://github.com/allanzelener/YAD2K))

#### Results

![1](https://github.com/ashishkg0022/YOLOv2-keras/blob/master/output_images/test_person.jpg)

![2](https://github.com/ashishkg0022/YOLOv2-keras/blob/master/output_images/test_cars_2.jpg)

![3](https://github.com/ashishkg0022/YOLOv2-keras/blob/master/output_images/test_giraffe.jpg)
