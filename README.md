# Explaining video classification model predictions

Given a video classification model and a video, return an explanation for the prediction.

## Getting started

### Prerequisites

This project utilises several feature extraction methods and different machine learning frameworks to create the explanations. Additionally, `lime_base` is needed in order to build the explanation model. See [here](https://github.com/marcotcr/lime) for LIME documentation.

```bash
$ pip install torch
$ pip install tensorflow
$ pip install clip
$ pip install transformers
$ pip install lime
```

### Models

This project contains a simple video classification network using during project development. The model was trained using [Pitt's dataset](https://people.cs.pitt.edu/~kovashka/ads/) to classify advertisement videos as exciting or not and funny or not. There are further instructions on setting up this model in the `basicModel/` directory if needed. This model was aggregated with videos from the [internet archive dataset](https://archive.org/detailstelevision) using active learning. The source code and further details can be found in `activeLearningModel.py`.


### Installing

Clone directory and travel to **explainability** directory.

```bash
$ git clone git@github.com:jovanneste/emotionClassificationFromVideos.git
$ cd emotionClassificationFromVideos/src/explainability/
```

## Deployment

### Command line deployment

```bash
$ python explain.py --video <video_path> --model <model_path> [--segments <int:number of segments>] [--features <int:number of features>] [--print <bool:show detailed outputs>]
```

### App

The explainability parts of this project were used in a demo paper submitted to ACM SIGAR Conference on Research and Development in Information Retrieval 2023. Screenshots for this paper were taken from a small `flask` app created for this purpose. The app can be run using the following commands.

```bash
$ cd app/
$ export FLASK_APP=run.py
$ flask run
```
## Project structure

Outline of project structure excluding the `data\` directory.

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
│   │   ├── mask.py
│   │   └── quantise_video.py
│   ├── helperFunctions
│   │   ├── splitVideos.py
│   │   ├── videoTimeDist.py
│   │   └── videoTimeDistribution.png
│   └── manual.md
└── timelog.md
```
