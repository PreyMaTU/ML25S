# Machine Learning 2025S

This is the repository for the Machine Learning course at TU Wien 
for the 2025S semester of group 17. Members of the group are:

- Christoph Neubauer
- Matthias Preymann
- Philipp Vanek

## Exercise 1

Four datasets were used to train three different classifiers to 
compare the performance, document differences in data preparation
and evaluate possible edge cases. To this end many different
configurations were used including grid search. The script operates as a CLI
app with multiple options. The following classifiers were implemented
using the scikit sklearn library:

- KNN
- Multilevel Neural Network
- Random Forests
- (Naive Bayes only simple test case)

The datasets are loaded from disk as ZIP files, or via `fetch_ucirepo`:

- Breast Cancer: The Breast Cancer dataset contains measurements of breast tissue. Each entry
  corresponds to a tumor sample and includes numerical values describing characteristics such as
  radius, texture, perimeter, and symmetry. The target variable is binary, indicating whether a
  patient has a "recurrence-event" or "no-recurrence-event". There are no missing values in this
  dataset, and it is moderately imbalanced, with roughly two-thirds of the samples being benign.

- Loan:  The Loan dataset contains financial information of applicants for a loan, where the target variable is
 the application score, ranging from the best score "A" to the worst "G". The target variable
 is rather normally distributed, with most records be in grad "B" or "C" (each 30%) and only a few
 records with "F" or "G".

- Dota: The Dota2 Games Results dataset captures the played games of Dota2 in a two hour timeframe
  and records which playable characters were picked, and which team finally won the game. This is the
  largest of our datasets with almost 100k entries. Apart from 3 categorical features (a clusterid,
  gamemode and gametype), all other features of one record represent one playable character each and
  capture if they were not played (0), or picked in team A (1) or team B (-1). The target is binary,
  with the possible outcomes of: Team A won (1) or Team B won (-1). The dataset is very sparse, as only
  10 characters (5 on each team) out of the 113 are played in each game.

- Heart Disease (via `fetch_ucirepo`): The Heart Disease dataset contains records of medical information
  about patients with the target variable of having a heart disease. The classes are "no disease" (0),
  which accounts for a little more than half of all records, and four different diseases (1-4), where
  some classes have only very few entries. Due to this reason, this dataset is often used for binary
  classification ("no disease" vs "disease present"), but we focus on the multi-class classification in
  our experiments, to keep the dataset in its original form. In later sections we give some observations
  on the differences in performance of the two versions.

Results can be exported and imported, so that the code for plotting can be
adapted without re-running the training and evaluation code.

## Exercise 2

Three implementations of multilayer neural networks were compared based on their
training and classification performance. The Heart Disease and Loan datasets from
the previous exercise 1 were used for training. The following networks were created:

- PyTorch Net: Regular classifier network using the PyTorch library.
  
- Scratch Net: A basic neural network classifier library written from scratch. Features
  a composable layer structure, different loss and activation functions, plus a few
  additional data processing utilities. Only relies on numpy and uses its vectorization.
  Has support for batch learning.

- LLM Generated Net: A purpose built classifier network for our two datasets only using
  numpy generated with ChatGPT based on a lengthy prompt. Needed only minor manual fixes.

## Exercise 3

A simplified version of the retro game Breakout was written and used to train
a reinforcement learning model. This uses on-policy first-visit Monte-Carlo control
(for epsilon-soft policies) with exponential epsilon-cooloff. The game can be run
interactively with the PyGame library and controlled using the arrow keys on the keyboard.
Else the game can be controlled using a trained policy and visualized with PyGame
or it can be run completely headlessly for training or evaluation.

## License

The code of this project is licensed under the MIT license.
