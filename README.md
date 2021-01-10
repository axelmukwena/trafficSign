# Instructions for running Files in the project 'trafficSign'

#### Make sure the following files/folders are in the same directory:
    - trafficSignTrain.py
    - trafficSignTest.py
    - TrafficSignData/Training

#### Run the trafficSignTrain.py
    - This file will output an image dataset sample
    - Accuracy scrores for cross validation and test scores based on trained data
    - also will export trained models for each classifeir as:
            - SVMmodel.pkl
            - GNBmodel.pkl
            - CLFmodel.pkl
        This files contain trained models of the classifiers and are loaded during
        and used during testing

#### Run the trafficSignTest.py
    - This file will output accuracy scores for tested models
    - This file is simply for testing the trained classifies and it'll load the trained
      models, and use them for classification