## Executables

- crawlVideos.py: Crawl the internet archive video and store the links to each commercial found
- downloadVideos.py: Download each of the videos found during the crawling
- annotateVideos.py: Annotate a small subset of the videos downloaded
- evaluateModel.py: Evaluate the current model on the 100 annotated videos (includes a cross-validation function)
- activeLearningModel.py: Random and uncertainty sampling in order to improve the current model

## Directories

- basicModel: Mostly code from Jin's project - trains a model on videos from Pitts dataset
- helperFunctions: Several helper functions to split and help understand the videos
- explainability: All files relating to making the model explainability
