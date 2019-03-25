# Quora-Question-Pair-Similarity

![hqdefault](https://user-images.githubusercontent.com/36497538/54881092-3b525300-4e72-11e9-84f7-9d1d0b35395a.jpg)

## Problem Statement
Identify which questions asked on Quora are duplicates of questions that have already been asked.
This could be useful to instantly provide answers to questions that have already been answered.
We are tasked with predicting whether a pair of questions are duplicates or not.

### Source : https://www.kaggle.com/c/quora-question-pairs 

## Real world/Business Objectives and Constraints 
- The cost of a mis-classification can be very high.
- You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
- No strict latency concerns.
- Interpretability is partially important.

## Data Overview 
- Data will be in a file Train.csv 
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate 
- Size of Train.csv - 60MB 
- Number of rows in Train.csv = 404,290

### Performance Metric 
Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s):
log-loss 
Binary Confusion Matrix

## To run on local browser
run app.py and open localhost/8080
![Capture](https://user-images.githubusercontent.com/36497538/54945082-05d56480-4f5b-11e9-982f-625e22594e7f.PNG)
