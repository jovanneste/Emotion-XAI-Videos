## Project plan


### 19/10/2022

- Read and make notes on new paper on active learning
- Look through references and find similar potentially useful papers
- Look through and run source code from masters project
- See what results I can and cannot reproduce
- Play with parameters
- Report on what I learned from paper and what parts of the project I could reproduce myself with the code

### 26/10/2022

- Download Pitts dataset
- See what results I can and cannot reproduce
- Play with parameters
- Research AL query based frameworks
- Read Active Learning Literature Survey

### 02/11/2022

- Find an optimal parameter configuration
- Test on my annotated data
- Start implementing uncertainty sampling

### 09/11/2022

- Time distribution histogram
- Add random sampling AL method
- Use cross validation on annotated videos
- Look into explanation methods for video classification

### 16/11/2022

- Look into explainability
- LIME and SHAP research
- Remove videos >60seconds

## Plan for Explainability

Retrieval task to find images that explain why a video has been labelled in some way

Given a model and synthetic videos (will make these and annotate using a boundary box) :



1. Find key frames from video
For each pair of frames:
- SSIM, or
- Sum of absolute difference thresholded
Or
- Scene detect (not necessarily a key frame)
- Average image over a given period of time (split video into n images)

2. Use LIME or SHAP on these key frames
Evaluate on synthetic videos

3. Compute overlap of LIME and SHAP pixel regions
Use some set similarity score (Jaccard) to order the key frames
Output some 'score' to represent how explainable the model is



Does model accuracy relate to model explainability  (is this novel?)

Update model to a joint model
Just ignore videos that the model labels 0,0
Does this just work on ad videos

Test data set would contain:
- Annotated ad videos
- Annotated synthetic videos


### 01/02/2023

- Explainability implemented
- Start small flask app
- Cherry pick examples for demo paper

## Plan for App

Small flask app with pre computed examples to show in demo paper
Use bootstrap with github 
