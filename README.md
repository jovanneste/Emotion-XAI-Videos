# Explaining video classification model predictions

Given a video classification model and a video, return an explanation for the prediction.

## Getting started

### Dependencies

This project utilises several feature extraction methods and different machine learning frameworks to create the explanations. Additionally, `lime_base` is needed in order to build the explanation model. See [here](https://github.com/marcotcr/lime) for LIME documentation. The required libraries can be installed using `pip`:

```bash
$ pip install torch
$ pip install tensorflow
$ pip install clip
$ pip install transformers
$ pip install lime
$ pip install shap
```

### Models

This project contains a simple video classification network using during project development. The model was trained using [Pitts dataset](https://people.cs.pitt.edu/~kovashka/ads/) to classify advertisement videos as exciting or not and funny or not. There are further instructions on setting up this model in the `basicModel/` directory if needed. This model was aggregated with videos from the [internet archive dataset](https://archive.org/detailstelevision) using active learning. The source code and further details can be found in `activeLearningModel.py`.

### Data

This repo omits the *data/* directory. To train the model, Pitts data should be downloaded and stored in this directory alongside the annotations: *video_Exciting_clean.json* and *video_Funny_clean.json*.


### Installing

Clone directory and travel to **explainability** directory.

```bash
$ git clone git@github.com:jovanneste/emotionClassificationFromVideos.git
$ cd emotionClassificationFromVideos/src/explainability/
```

## Deployment

### Command line deployment

`explain.py` requires a video file and a Keras model and returns an explanation.

```bash
$ python explain.py --video <> --model <> [--segments <int>] [--features <int>] [--print <bool>]
```

- `segments` : how many super pixel regions to segment our top frame into,
- `features` : how many pixel regions to show in output,
- `print` : when true, system will output step-by-step details.  

Argument descriptions can be found using "--help":

```bash
$ python explain.py --help
```

### App

The explainability parts of this project were used in a demo paper submitted to ACM SIGAR Conference on Research and Development in Information Retrieval 2023. Screenshots for this paper were taken from a small `flask` app created for this purpose. The app can be run using the following command:

```bash
$ cd app/
$ export FLASK_APP=run.py
$ flask run
```
## Project structure

Outline of project structure excluding the `data/` directory.

```bash
.
├── README.md
├── app
│   ├── run.py
│   ├── static
│   │   ├── scripts.js
│   │   └── styles.css
│   └── templates
│       └── index.html
├── plan.md
├── research
├── src
│   ├── README.md
│   ├── activeLearningModel.py
│   ├── annotateVideos.py
│   ├── basicModel
│   │   ├── downloadVideos.py
│   │   ├── sampleVideos.py
│   │   └── trainModel.py
│   ├── crawlVideos.py
│   ├── downloadVideos.py
│   ├── evaluateModel.py
│   ├── explainability
│   │   ├── explain.py
│   │   ├── lime_video.py
│   │   ├── shap_video.py
│   │   ├── mask.py
│   │   └── quantise_video.py
│   ├── helperFunctions
│   │   ├── splitVideos.py
│   │   ├── videoTimeDist.py
│   │   └── videoTimeDistribution.png
│   └── manual.md
└── timelog.md
```
