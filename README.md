8-bit Bias Bounty Entry
===================================

This is a pretrained EfficientNet v2 large model fine-tuned solely on the `train` dataset of the [8-bit Bias Bounty competition](https://biasbounty.ai/8-bbb). It outputs a probability vector of size 17 for each image, where the extra dimension is used to predict whether or not an image is out-of-distribution, i.e. not a person. Inference with this model outputs both these probabilities and the text labels the top-1 probabilites imply.

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

This is assuming you're on a Linux machine with a CUDA-enabled GPU and `conda`.

You should be able to recreate the environment I used for training and inference with `conda env create -f environment.yml`.

If something there goes wrong, these were the commands used to setup the environment:
```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c conda-forge matplotlib
pip install Pillow scikit-learn tqdm tensorboard
pip install dlib face_recognition
pip install opencv-python pyyaml seaborn optuna plotly
```

Unzip Data
----------

This project assumes a subdirectory named `raw_data` with two subdirectories: `train` and `test`. The `train` subdirectory is required if models are going to be trained, and the `test` subdirectory is required if predictions are going to be made. It assumes each of these subdirectories contain `labels.csv` file and a bunch of images. This layout comes directly from the zip dataset provided. (This layout is only assumed in the Makefile. Changing the Makefile to point to other directories should work.)

Preprocessing
-------------

Instead of relying on the model to perform randomly on ImageNet images, we'll predict whether an image is out-of-distribution. To do this, we need labels for which images from the training set are likely from ImageNet. To create these labels, we use a combination of YOLOv5 and dlib (via the face_recognition package) to add extra label columns to the `labels.csv` file. To run this step, run `make data`.

We then manually review the labels and write out our changes. This takes place in the "Find ImageNet Images" section of `notebook.ipynb`.

We also created weights for each of the classes to handle the unbalancedness (and as a naive approach to addressing disparity in accuracy). This is in the "Create Class Weights" section of `notebook.ipynb`.

Training
--------

Training is setup to automatically run hyperparameter tuning. The parameter space is just defined in `train.py`. Once these are tweaked, training can be run with `make train`. Tensorboard logs will be dumped to `models/tensorboard`. Models will be spit out to the `models` directory. Training takes roughly one minute per epoch on an RTX 2070 Super.

We were lazy and hard-coded some paths in `train.py`, so you'll need to edit those path constants at the top of the file.

Predictions
-----------

Run `make predictions`. You may need to alter the model ID or dataset location in the Makefile to suit your needs. This will output three files into `predictions/`:
* One with class probabilities (`{model_id}_probs.pkl`)
* One with text labels (`{model_id}_labels.pkl`)
* One with labels as a dataframe (`{model_id}_labels.csv`)

We were lazy and hard-coded some paths in `predict.py`, so you'll need to edit those path constants at the top of the file.

Assess Performance
------------------

An example of performance assessment is in the "Assess Predictions" section of `notebook.ipynb`. We load a dataset, load the predictions for a model id, and compare them. We compare them using the `_probs.pkl` file, but one of the `_labels` files is probably more convenient.

Feedback for Competition Organizers
-----------------------------------

1. Please don't provide labels we aren't supposed to train on. I didn't use them because you asked that we not, but model performance on the private dataset would have improved if I had. You also don't have a way of verifying that the competitors didn't train on them. We're strongly incentivized to use them, and the competitors that obey the rules and don't use them end up with worse performing models.
2. Are you considering the randomly assigned ImageNet labels in final accuracy scores in the same way they're being considered in the public leaderboard? I don't think you should. Your scoring metric is fairly sensitive to accuracy, especially with skin tone, and simply by chance some people will get more of these correct. These people could be pushed higher on the leaderboard than those with more correct predictions *on the observations with real labels*, which doesn't seem to be in line with what you want to measure. 
3. The scoring rubric ignores intersectionality. If I have low disparity between skin tones and low disparity between perceived genders, I could still have high disparity between e.g. white men and black women. This goes for all the 3-tuples as well (e.g. white, old men).
4. Asking that we submit a Github repo with a model in it is a bit unrealistic due to the (often) large file sizes involved. For this reason, best practice guides often discourage committing models. Consider better submission guidelines around this, or don't require the submission of models at all (just the submission of predictions).
5. Similarly, the competition didn't ask for any standard format for outputs or predictions or anything. My spouse works for a data science competition platform, so I've seen what this looks like from the other side, and you could have been in for a world of hurt if a ton of people had taken part. You've signed yourself up for getting every single repo running and outputting predictions, which sounds like one of the circles of hell. I'm worried this won't scale for future competitions.
6. The topic of this competition obviously triggers a strong reaction in most people, and the only reason I considered taking part is because I've spoken to Rumman previously and asked her a couple questions about this. This is this organization's first competition, so trust from the community is relatively low. Taking on such a controversial topic without building that trust first might have limited the potential of this competition.