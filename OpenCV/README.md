
# *OpenCV* - Udemy course

This repository contains Machine learning tutorials and practice codes from the Udemy course 
**Machine Learning A-Z™: Hands-On Python & R In Data Science**.

## Contents in the Repository
1. [About the Course](#1-about-the-course)
2. [What's in the Course](#2-whats-in-the-course)
3. [Requirements](#3-requirements)
3. [Installation](#4-installation)


## 1. About the Course (Description from creator)

Welcome to the ultimate online course on Python for Computer Vision!

This course is your best resource for learning how to use the Python programming language for Computer Vision.

We'll be exploring how to use Python and the OpenCV (Open Computer Vision) library to analyze images and video data.

The most popular platforms in the world are generating never before seen amounts of image and video data. Every 60 seconds users upload more than 300 hours of video to Youtube, Netflix subscribers stream over 80,000 hours of video, and Instagram users like over 2 million photos! Now more than ever its necessary for developers to gain the necessary skills to work with image and video data using computer vision.

Computer vision allows us to analyze and leverage image and video data, with applications in a variety of industries, including self-driving cars, social network apps, medical diagnostics, and many more.

As the fastest growing language in popularity, Python is well suited to leverage the power of existing computer vision libraries to learn from all this image and video data.

In this course we'll teach you everything you need to know to become an expert in computer vision! This $20 billion dollar industry will be one of the most important job markets in the years to come.

We'll start the course by learning about numerical processing with the NumPy library and how to open and manipulate images with NumPy. Then will move on to using the OpenCV library to open and work with image basics. Then we'll start to understand how to process images and apply a variety of effects, including color mappings, blending, thresholds, gradients, and more.

Then we'll move on to understanding video basics with OpenCV, including working with streaming video from a webcam.  Afterwards we'll learn about direct video topics, such as optical flow and object detection. Including face detection and object tracking.

Then we'll move on to an entire section of the course devoted to the latest deep learning topics, including image recognition and custom image classifications. We'll even cover the latest deep learning networks, including the YOLO (you only look once) deep learning network.

This course covers all this and more, including the following topics:

  * NumPy
  * Images with NumPy
  * Image and Video Basics with NumPy 
  * Color Mappings
  * Blending and Pasting Images
  * Image Thresholding
  * Blurring and Smoothing
  * Morphological Operators
  * Gradients 
  * Histograms 
  * Streaming video with OpenCV 
  * Object Detection 
  * Template Matching 
  * Corner, Edge, and Grid Detection 
  * Contour Detection 
  * Feature Matching 
  * WaterShed Algorithm 
  * Face Detection 
  * Object Tracking 
  * Optical Flow 
  * Deep Learning with Keras 
  * Keras and Convolutional Networks 
  * Customized Deep Learning Networks 
  * State of the Art YOLO Networks 
  * and much more!

Feel free to message me on Udemy if you have any questions about the course!
Jose

More about the course can be found [here](https://www.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/) 


## 2. What's in the Course

 * Understand basics of NumPy
 * Manipulate and open Images with NumPy
 * Use OpenCV to work with image files
 * Use Python and OpenCV to draw shapes on images and videos
 * Perform image manipulation with OpenCV, including smoothing, blurring, thresholding, and morphological operations.
 * Create Color Histograms with OpenCV
 * Open and Stream video with Python and OpenCV
 * Detect Objects, including corner, edge, and grid detection techniques with OpenCV and Python
 * Create Face Detection Software
 * Segment Images with the Watershed Algorithm
 * Track Objects in Video
 * Use Python and Deep Learning to build image classifiers
 * Work with Tensorflow, Keras, and Python to train on your own custom images.


## 3. Requirements

* python 3.7.6
* cuda 10.1
* cudnn 7.6.5

  ```
   # to check cudnn version on windows
   Go to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\
   open cudnn.h
   define CUDNN_MAJOR 7
   define CUDNN_MINOR 6
   define CUDNN_PATCHLEVEL 5
   in my case its 7.6.5
  ```

## 4. Installation 

1. Install CUDA 10.1 and CUDNN 7.6.5

2. Create python virtual environment with pip.

   ```bash
   pip install virtualenv
   cd project_folder
   source venv/bin/activate
   ```
3. PIP install the following libraries:
   * OpenCV
   * Pandas
   * Scikit Learn
   * PIL


