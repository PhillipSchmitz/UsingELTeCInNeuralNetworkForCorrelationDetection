# Project

The goal of this project was to check whether attributes in a pre-modified ELTeC dataset correlate with one another when fed into a neural network to predict different target attributes.
The neural network used was implemented by Samson Zhang (https://www.kaggle.com/wwsalmon) a few years ago. 

# Goal

If attributes are missing within the dataset fed to the neural network there may or may not be a decrease in accuracy, depending on their correlations with others. We want to identify those correlations and dependencies between attributes.
In a positive case we detect a lower score indicating those excluded attributes must be important for the model to output a better value. A combination of all correlating attributes must then lead to an excellent performance of the network.

# Methodology

A target attribute is selected first and one attribute out of all remaining attributes is selected and excluded from the dataset that is used as input for the neural network. For each attribute several runs are undertaken to best compare the overall performance.
The experiment ends when every attribute has been exluded once and the different performance values have been checked and compared. Ideally, the attributes that strongly correlate with others can be combined to ultimately lead to the best overall performance score.

# Background

The topic was proposed in the digital humanities master program at Trier University and falls under a mandatory term paper to pass the respective subject as a student. Any findings are not final and may be used for further work, if needed.
