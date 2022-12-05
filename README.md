8-bit Bias Bounty
=================

Usage Overview
--------------
1. [Install](#install-requirements) the requirements.
2. [Unzip data](#unzip-data) to the appropriate directory.
3. [Preprocess](#preprocessing) the data.
3. [Train](#training) the model.
4. [Make predictions](#predictions) on test data.
5. [Assess](#assess-performance) model performance.


Install Requirements
--------------------

This is assuming you're on a Linux machine with a CUDA-enabled GPU.

You should be able to recreate the environment I used for training and inference with `conda env create -f environment.yml`.

Unzip Data
----------

This project assumes a subdirectory named `raw_data` with two subdirectories: `train` and `test`. The `train` subdirectory is required if models are going to be trained, and the `test` subdirectory is required if predictions are going to be made. It assumes each of these subdirectories contain `labels.csv` file and a bunch of images. This layout comes directly from the zip dataset provided. (This layout is only assumed in the Makefile. Changing the Makefile to point to other directories should work.)

Preprocessing
-------------

Instead of relying on the model to perform randomly on ImageNet images, we'll predict whether an image is out-of-distribution. To do this, we need labels for which images from the training set are likely from ImageNet. To create these labels, we use a combination of YOLOv5 and dlib (via the face_recognition package) to add extra label columns to the `labels.csv` file. To run this step, run `make data`.

We then manually review the labels and write out our changes. This takes place in the "Find ImageNet Images" section of `notebook.ipynb`.

Training
--------

Training is setup to automatically run hyperparameter tuning. The parameter space is just defined in `train.py`. Once these are tweaked, training can be run with `make train`. Tensorboard logs will be dumped to `models/tensorboard`. Models will be spit out to the `models` directory. Training takes roughly one minute per epoch on an RTX 2070 Super.

Predictions
-----------

Run `make predictions`. 

Assess Performance
------------------

Feedback for Competition Organizers
-----------------------------------

* It didn’t say there were missing labels
* Why did we get the test labels? You request that we only train on train, but will you know if we improve performance by training on train and test?
* The scoring rubric is ignoring intersectionality
* Controversial topic without building trust first
* No one checks models into git repos
* No standardization of model outputs.
