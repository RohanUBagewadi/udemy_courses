
# *Udemy Courses*

This repository contains all the purchased tutorials and practice codes from the Udemy courses.

## Contents in the Repository
1. [List of Courses](#1-list-of-courses)
3. [Requirements](#3-requirements)
3. [Installation](#4-installation)


## 1. List of Courses

* [Machine Learning with Python](Machine_Learning_Python)

## 2. Requirements

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

## 3. Installation 

1. Install CUDA 10.1 and CUDNN 7.6.5

2. Create python virtual environment with pip.

   ```bash
   pip install virtualenv
   cd project_folder
   source venv/bin/activate
   ```
3. PIP install the following libraries:
   * Pandas
   * Scikit Learn
   * OpenCV
   * PIL


