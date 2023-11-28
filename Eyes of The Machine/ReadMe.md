# GeoPACHA: Eyes of the Machine --  AI-assisted Satellite Archaeological Survey in the Andes

This repository contains scripts, workflows, the model bundle, and other relevant information for understanding and implementing the AI-assisted satellite survey conducted for this paper. Further information about these files is provided below.

## PreChippedClassification
Usually, Rastervision creates training chips by defining an area of interest and creating a sliding window that extracts chips of a defined size across that area of interest. (Or random windows of a defined size within the AOI) However, our training and validation data is composed of randomly distributed chips across the survey region. The chip boundaries are defined by polygons stored in geojson files. This code defines a plugin for RasterVision that uses the chip polygons to extract image chips.

## Processed (Training/Validation Data)
Unfortunately, the imagery is too large to be distributed here, however, the files in the "Processed" directory contain the rest of the data used in the training and evaluation of the model.

### Scenes
Scenes contains CSV files that RasterVision uses to create "Scenes." Scenes combine togther the Imagery, Training data, and any Areas of Interest to make a single data object with all relevant information. This is then used to extract image chips and create datasets to train the model.

### TrainingGeoJSON
GeoJSON files which contain the polygons for the Training data. The polygons in these files are aligned with the particular imagery we used in this analysis, which means they are likely close to aligned with publicly available google earth/bing imagery, but may not be slightly shifted in some cases.

### ValidationGeoJSON
GeoJSON files which contain the polygons for the Validation data. The polygons in these files are aligned with the particular imagery we used in this analysis, which means they are likely close to aligned with publicly available google earth/bing imagery, but may not be slightly shifted in some cases.


## Rastervision Outputs

### Model Bundle
The model bundle is a zip file (named model-bundle.zip) produced by RasterVision which contains the model weights and all of the information necessary to deploy the model on new imagery. It also contains log files detailing the configurations used in the production of the model.

Within model-bundle.zip is pipeline-config.json, which contains all of the configurations used in training the model, and another zip file, also called model-bundle.zip. This second model-bundle.zip contains a pipeline-config.json file containing all of the configurations necessary to deploy the model, and a model.pth file which contains the model weights in a pytorch readable format.

### Train Directory
- **dataloaders** - sample images from training, valiation and testing sets showing that the data loaders are behaving properly
- **tb-logs** - Logs of training viewable in Tensorboard to track training progress and troubleshoot problems
- last-model.pth - model weights at the end of training
- learner-config.json - configuration file showing the setup used in model training
- log.csv - Relevant statistics about each training epoch
- model-bundle.zip -  a copy of model-bundle discussed above which can be used to continue training if desired
- test_metrics.json - metrics of model evaluation in json format
- test_preds.png - examples of predictions made by the model for randomly selected chips

### Makefile
The rastervision commands that run the pipeline. THis file is produced by rastervision as a part of model production.

### pipeline-config.json
The pipeline configuration in json format

## ArchaeoStruct_Pipeline2.py
Python code to run the Rastervision pipeline and define the configurations.

## RunningRastervision.txt
A quick and dirty text file contiaining commandline code that is useful for running Rastervision from Docker.