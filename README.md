# Project

The idea for this project was given in the course of a seminar in the digital humanities master program at Trier University. It was part of an academic term paper.
The original neural network used was implemented by Samson Zhang (https://www.kaggle.com/wwsalmon) a few years ago.

# Goal
We want to make use of Zhang's neural network and see how it performs on the ELTeC dataset(s) (https://github.com/dh-trier/datasets/tree/main/tabular/eltec) that consists of word counts of words (features) a total of ten authors used in the book they wrote. The features are used to train and evaluate the network's classification accuracy. With the use of Backward Elimination, we want to reduce the number of features, but increase the model's performance.


# Methodology
The number of features in the dataset(s) are quite big and would thereby cause high computation times if used for training the network.
To avoid this scenario, the first 500 features are used to run experiments with the remaining parameters (generally: train/dev splits: 60/40, learning rate: 0.1, iteration (gradient descent): 500; 1000, 2000 (optional)).
A newly added function that implements a basis Backward Elimination approach, with first smaller refinements, generates the best features and model accuracy. The script computes and stepwise saves the relevant information and files for further analysis and interpretation.
