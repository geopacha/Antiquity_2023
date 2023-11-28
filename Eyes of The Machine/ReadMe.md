# GeoPACHA: Eyes of the Machine --  AI-assisted Satellite Archaeological Survey in the Andes

This repository contains scripts, workflows, the model bundle, and other relevant information for understanding and implementing the AI-assisted satellite survey conducted for this paper.

## Imagery Preprocessing
## Model Bundle
The model bundle is a zip file (named model-bundle.zip) produced by RasterVision which contains the model weights and all of the information necessary to deploy the model on new imagery. It also contains log files detailing the configurations used in the production of the model.

Within model-bundle.zip is pipeline-config.json, which contains all of the configurations used in training the model, and another zip file, also called model-bundle.zip. This second model-bundle.zip contains a pipeline-config.json file containing all of the configurations necessary to deploy the model, and a model.pth file which contains the model weights in a pytorch readable format.

