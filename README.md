# Explaining video classification model predictions

Given a video classification model and a video, return an explanation for the prediction.

## Getting started

### Prerequisites

This project utilises several feature extraction methods and different machine learning frameworks to create the explanations. Additionally, *lime_base* is needed in order to build the explanation model.

```bash
$ pip install torch
$ pip install tensorflow
$ pip install clip
$ pip install transformers
$ pip install lime
```

### Models

This project contains a simple video classification network using during project development. The model was trained using [Pitt's dataset](https://people.cs.pitt.edu/~kovashka/ads/) to classify advertisement videos as exciting or not and funny or not. There are further instructions on setting up this model in the **basicModel** directory if needed. This model was aggregated with videos from the [internet archive dataset](https://archive.org/detailstelevision) using active learning. The source code and further details can be found in **activeLearningModel.py**.


### Installing

Clone directory and travel to **explainability** directory.

```
$ git clone git@github.com:jovanneste/emotionClassificationFromVideos.git
$ cd emotionClassificationFromVideos/src/explainability/
```


## Deployment



<!-- Builds on work done by Jin Wang (Classifying Emotion Types in Advertisement Videos by a Deep Learning Approach)

## Results

Spreadsheet of [results](https://docs.google.com/spreadsheets/d/1t6kJvhNDyAogelgcSPBLtxTqOguTWPabCLRDLFe8rEc/edit?usp=sharing) -->
