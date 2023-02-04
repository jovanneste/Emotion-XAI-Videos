## Initial Model set-up (basicModel)

- Download Pitts dataset (including annotations)
- Run **trainModel.py**

## Run Model on new data

- Crawl and download videos from Internet Archive
- Split videos into test and train
- Annotate the test set
- Run **evaluateModel.py** to run model on test videos (cross-validation available)

## Improve Model

- Run **activeLearningModel.py** with either random or uncertainty sampling

## Explain model

- Run **explainModel.py** with a video and a keras model which outputs the most important frame with most imporatnt pixel regions highlighted 
