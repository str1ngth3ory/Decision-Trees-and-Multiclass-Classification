# CS 6601: Artificial Intelligence - Assignment 4 - Decision Trees and Multiclass-Classification

## Setup
Clone this repository:
```
git clone https://github.gatech.edu/omscs6601/assignment_4.git
```
You will be able to use Numpy for the assignment 

You will be able to use sklearn and graphviz for jupyter notebook visualization only

No other external libraries are allowed for solving this problem.

Please use the ai_env environment from previous assignments

If you are using an IDE:
Ensure that in your preferences, that ai_env is your interpreter and that 
the assignment_4 directory is in your project structure

Else: From your ai_env Terminal:
```
conda activate ai_env
```
The supplementary testing notebooks use jupyter:
visualize_tree and unit_testing. From your ai_env Terminal:
```
jupyter notebook
```
The supplementary Helper notebook to visualize Decision tree, 
requires the graphviz library. From your ai_env Terminal:
```
pip install graphviz==0.17.0
or alternatively
pip install -r requirements
```
If you have difficulty or errors on graphviz0.17 
From your ai_env Terminal:
```
conda install -c conda-forge python-graphviz 
```
which installs version 0.19 (compatible)


## Overview
Machine Learning is a subfield of AI, and Decision Trees are a type of Supervised Machine Learning. In supervised 
learning an agent will observe sample input and output and learn a function that maps the input to output. The function
is the hypothesis ''' y = f(x)'''. To test the hypothesis we give the agent a *test set* different than the training 
set. A hypothesis generalizes well if it correctly predicts the y value. If the value is *finite*, the problem is a 
*classification* problem, if it is a *real number* it is considered a *regression* problem.
When classification problems have exactly two values (+,-) it is Boolean classification. When there are more than 
two values it is called Multi-class classification.

Decision trees are relatively simple but highly successful types of supervised learners. Decision trees take a vector of
attribute values as input and return a decision.

<p> <img src="./files/dt.png" alt="Decision Trees" width="700" height="350"/>

## Submission and Due Date

The deliverable for the assignment is to upload a completed **_submission.py_** to Gradescope.

* All functions must be completed in **_submission.py_** for full credit

**Important**:
Submissions to Gradescope are rate limited for this assignment. **You can submit two submissions every 60 minutes during the duration of the assignment**.

Since we want to see you innovate and imagine new ways to do this, we know this can also cause you to fail 
(spectacularly in my case) For that reason you will be able to select your strongest submission to Gradescope. 
In your Gradescope submission history, you will be able to mark your best submission as 'Active'. 
This is a students responsibility and not faculty.

### The Files

You are only required to edit and submit **_submission.py_**, but there are a number of important files:
1. **_submission.py_**: Where you will build your decision tree, confusion matrix, performance metrics, forests, and do the vectorization warm up.
2. **_decision_trees_submission_tests.py_**: Sample tests to validate your trees, learning, and vectorization locally.
3. **_visualize_tree.ipnb_**: Helper Notebook to help you understand decision trees of various sizes and complexity
4. **_unit_testing.ipynb_**: Helper Notebook to run through tests sequentially along with the readme

### Resources
* Canvas *Thad's Videos*: [Lesson 7, Machine Learning](https://gatech.instructure.com/courses/225196/modules/items/2197076)
* Textbook:<br> 
  * Chapter 18 Learning from Examples<br>
  * Chapter 20 Learning Probabilistic Models
* [Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
* [K-Fold Cross-validation](http://statweb.stanford.edu/~tibs/sta306bfiles/cvwrong.pdf)

### Decision Tree Datasets
1. **_part23_data.csv_**: 4 features, 1372 data points, binary classification (last column)
2. **_challenge_train.csv_**:  30 features, 6636 datapoints, binary classification (first column)
3. **_challenge_test.csv_**: (will not be provided, but will be similarly structured as challenge_train with 40% of the datapoints)

#### Warmup Data
4. **_vectorize.csv_**: data used during the vectorization warmup for Assignment 4

### Imports
**NOTE:** We are only allowing three imports: numpy, collections.Counter and time. We will be checking to see if any other libraries are used. You are not allowed to use any outside libraries especially for part 4 (challenge). Please remember that you should not change any function headers.

---

### Part 0: Vectorization!
_[6 pts]_

* File to use: **_vectorize.csv_**

The vectorization portion of this assignment will teach you how to use matrices to significantly increase the speed
and decrease processing complexity of your AI problems. Matrix operations, linear algebra and vector space axioms and
operations will be challenging to understand. Focus on this, and realize it will benefit you throughout AI courses and your career.

You will not be able to meet time requirements, nor process large datasets without Vectorization
Whether one is training a deep neural network on millions of images, building random forests over a large dataset, or 
utilizing other algorithms, machine learning uses _extensive_ use of vectorization. In python, the **numpy** package 
provides a programmer with the ability to use python-wrapped, low-level optimizations written in C.

Data management in Assignment 4 will benefit from familiarity with these techniques. It also has a vectorization 
requirement to run within a reasonable time limit. This small section will introduce you to vectorization and 
some of the cool tricks you can use in python. We encourage you to use any numpy function out there to do the 
functions in the warmup section.

You will need to beat the non-vectorized code to get full points.

TAs will not help on this section. This section was created to help get you ready for this and other assignments; 
feel free to ask other students on Ed Discussion or use the Internet.

How grading works:
1. We run the non-vectorized code and your vectorized code 500 times, as long as the average time of your vectorized code is less than the average time of the non-vectorized code, you will get the points (given that your answer is correct).

#### Functions to complete in the `Vectorization` class:
1. `vectorized_loops()`
2. `vectorized_slice()`
3. `vectorized_flatten()`

---

## The Assignment
E. Thadeus Starner is the 3rd incarnation of the great innovator and legendary pioneer of Starner Eradicatus 
Mosquitoes. For centuries the mosquito has imparted only harm on human health, aiding in transmission of malaria, dengue, 
Zika, chikungunya, CoVid, and countless other diseases impact millions of people and animals every year. The Starner
Eradicatus, *Anopheles Stephensi* laser zapper has obtained the highest level of precision, recall, and accuracy
in the industry!

The secret is in the classification engine which has built an unmatched library of classification data collected from 
153 countries. Flying insects from the tiny Dicopomorpha echmepterygis (Parasitic Wasp) to the giant titanus giganteus 
(Titan Beetle) are care catalogued to a comprehensive digital record and indexed to support fast and correct classification.
This painstaking attention to detail was ordered by A. Thadeus (1st) to address a tumultuous backlash from the 
International Pollinators Association to a high mortality among beneficial pollinators.

E. Thadeus' close friend Skeeter Norvig, a former CMAO (Chief Mosquito Algorithm Officer) and pollinator advocate has 
approached E.T. with an idea. The agriculture industry has been experiencing terrible losses worldwide due to the 
diaphorina citri (Asian Citrus Psyllid), drosophila suzuki (spotted wing Drosophila), and the bactrocera tyron 
(Queensland fruit fly). Wonderful! E.T. exclaims, and becomes wildly excited at the opportunity to bring such an 
important benefit to the World.

The wheels of invention lit up the research Scrum that morning as E.T. and Skeeter storyboard the solution. People are
calling out all the adjustments, wing acoustics, laser power and duration, going through xyz positioning, angular 
velocity and acceleration calculations, speed, occlusion noise and tracking errors. You as the lead DT software 
engineer are taking it all in, when you realize and speak up..., sir... Sir... SIR... and a hush falls. Sir, we are 
doing Boolean classification and will need to move to multiclass classification. E.T. turns to you and with that look
in his eye, gives you and your team two weeks to deliver multiclass classification!

You will build, train and test decision tree models to perform multi-class classification tasks. You will learn 
how decision trees and random forests work. This will help you develop an intuition for how and why accuracy differs 
for training and testing data based on different parameters.

### Assignment Introduction
For this assignment we need an explicit way to make structured decisions. The `DecisionNode` class will be used to 
represent a decision node as some atomic choice in a multi-class decision graph. You must use this implementation
for the nodes of the Decision Tree for this assignment to pass the tests and receive credit.

An object of type 'DecisionNode' can represent a 
* decision node
  * *left*: will point to less than or equal values of the split value, type DecisionNode, True evaluations
  * *right*: will point to greater than values of the split value, type DecisionNode, False evaluations
  * *decision_function*: evaluates an attribute's value and maps each vector to a descendant
  * *class_label*: None
* leaf node
    * *left*: None
    * *right*: None
    * *decision_function*: None
    * *class_label*: A leaf node's class value
* Note that in this representation 'True' values for a decision take us to the left.

---

### Part 1a: Building a Binary Tree by Hand
_[5 Pts]_

In `build_decision_tree()`, construct a decision tree capable of predicting the class (col y) of each row. 
Using the columns A0-A3 build the decision tree and nodes in python to classify the data with 100% accuracy. 
Your tests should use as few attributes as possible, break ties with the equal select attributes by selecting
the one which classifies the greatest number of examples correctly. For ties in both attributes and correct classifications
use the lower index numbers (e.g. select **A1** over **A2**)
<p>

|  X  |   A0    |   A1    |   A2    |   A3    |  y  |
| --- | ------- | ------- | ------- | ------- | --- |
| x01 |  1.1125 | -0.0274 | -0.0234 |  1.3081 |  1  |
| x02 |  0.0852 |  1.2190 | -0.7848 | -0.7603 |  2  |
| x03 | -1.1357 |  0.5843 | -0.3195 |  0.8563 |  0  |
| x04 |  0.9767 |  0.8422 |  0.2276 |  0.1197 |  1  |
| x05 |  0.8904 | -1.7606 |  0.3619 | -0.8276 |  0  |
| x06 |  2.3822 | -0.3122 | -2.0307 | -0.5065 |  2  |
| x07 |  0.7194 | -0.4061 | -0.7045 | -0.0731 |  2  |
| x08 | -2.9350 |  0.7810 | -2.5421 |  3.0142 |  0  |
| x09 |  2.4343 | -1.5380 | -2.7953 |  0.3862 |  2  |
| x10 |  0.8096 | -0.2601 |  0.5556 |  0.6288 |  1  |
| x11 |  0.8577 | -0.2217 | -0.6973 | -0.1095 |  1  |
| x12 |  0.0568 |  0.0696 |  1.1153 | -1.1753 |  0  |


#### Requirements:
The total number of elements(nodes, leaves) in your tree should be < 10

#### Hints:
To get started, it might help to **draw out the tree by hand** with each attribute representing a node.

To create the decision function that will be passed to `DecisionNode`, you can create a lambda expression as follows:
'''func = lambda feature : feature[2] <= 0.356'''

This will choose the left node if the A2 attribute is 0.

For example, a hand binary tree might look like this:
> if A0<=-0.918 then class = 1; else class = 0
> <p>
> <img src="./files/tree_example.png" alt="Tree Example"/>

You would write your code like this:

    decision_tree_root = DecisionNode(None, None, lambda feature: feature[0] <== -0.918)
    decision_tree_root.left = DecisionNode(None, None, None, 1)
    decision_tree_root.right = DecisionNode(None, None, None, 0)

    return decision_tree_root

#### Functions to complete in the `submission` module:
1. `build_decision_tree()`

---

### Part 1b: Precision, Recall, Accuracy and Confusion Matrix
_[12 pts]_

To build the next generation Starner Zapper, we will need to keep the high levels of Precision, Recall, and Accuracy
inculcated in the legacy products. So you must find a new way, in binary or boolean classification we find these metrics
in terms of true positives, false positives, true negatives, and false negatives. So it should be simple right?
* Precision: Out of all the examples identified belonging to the class, how good were my clf predictions? TP/(TP + FP)
* Recall: Out of all the examples that belonged to the class, how many did my clf predict? TP/(TP + FN)
* Accuracy: Out of all the examples, how many did my clf predict correctly? (TP + TN)/(TP + TN + FP + FN)

To evaluate a learner we usually reserve a portion of the data for training and another portion for testing, 
There are many benefits to using cross-validation. See book 3rd Ed Evaluating and Choosing the Best Hypothesis 18.4 
Fill out the methods to compute the confusion matrix, accuracy, precision and recall for your classifier output. 
`classifier_output` will be the labels that your classifier predicts, while the `true_labels` will be the true test labels. 
You can refer to [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix) for help.

If you want to calculate the example set above by hand, run the following code.

    classifier_output = [decision_tree_root.decide(example) for example in examples]

    p1_confusion_matrix = confusion_matrix(classifier_output, classes)
    p1_accuracy = accuracy( classifier_output, classes )
    p1_precision = precision(classifier_output, classes)
    p1_recall = recall(classifier_output, classes)

    print p1_confusion_matrix, p1_accuracy, p1_precision, p1_recall

#### Functions to complete in the `submission` module:
1. `confusion_matrix()`
2. `precision()`
3. `recall()`
4. `accuracy()`

---

### Part 2a: Decision Tree Learning
_[10 pts]_

Purity, we strive for purity, alike Sir Galahad the Pure... 
As I am sure you have noticed, splitting at a decision node is all about purity. You are trying to improve information
gain which means you are trying to gain purer divisions of the data. Through purer divisions of the data it is more 
ordered, which relates to entropy in physics. Ordered motion produces more energy. Through ordered data you gain more 
information on the defining characteristics (attributes) of ßsomething observed.

We will use GINI impurity and Impurity Index to calculate the  `gini_impurity` and `gini_gain()` on the splits to 
calculate Information Gain. The challenge will be to choose the best attribute at each decision with the lowest 
impurity and the highest index. At each attribute we search for the best value to split on, the hypotheses are compared
against what we currently know, because would we want to split if we learn nothing?
Hints: 
* [gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
* [information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) 
* The Gini Gain follows a similar approach to information gain, replacing entropy with Gini Impurity.
* Numpy helpful capabilities such as binary masks, filtering arrays with masks, slicing, stacking and concatenating

<p>

#### Functions to complete in the `submission` module:
1. `gini_impurity()`
2. `gini_gain()`

---

### Part 2b: Decision Tree Learning
_[20 pts]_
* Data to train and test with: **_simple_multi.csv, mod_complex_multi_**
* Grading: average test accuracy over 10 rounds should be >= 70%
Meanwhile in the lab...
As the size of our flying training set grows, it rapidly becomes impractical to build multiclass trees by hand. 
We need to add a class with member functions to manage this, it is too much! To do list:
* Initialize the class with useful variables and assignments
* Fill out the member function that will fit the data to the tree, using build
* Fill out the build function
* Fill out the classify function

For starters, consider these helpful hints for the construction of a decision tree from a given set of examples:
1. Watch your base cases:
   1. If all input vectors have the same class, return a leaf node with the appropriate class label.
   2. If a specified depth limit is reached, return a leaf labeled with the most frequent class.
   3. Splits producing 0, 1 length vectors
   4. Splits producing less or equivalent information
   5. Division by zero
2. Use the DecisionNode class
3. For each attribute alpha: evaluate the information gained by splitting on the attribute `alpha`.
4. Let `alpha_best` be the attribute value with the highest information gain.
5. As you progress in this assignment this is going to be tested against larger and more complex datasets, think about how it will affect your identification and selection of values to test.
6. Create a decision node that splits on `alpha_best` and split the data and classes by this value.
7. When splitting a dataset and classes, they must stay synchronized, do not orphan or shift the indexes independently
8. Use recursion to build your tree, by using the split lists, remember true goes left using decide
9. Your features and classify should be in numpy arrays where for dataset of size (_m_ x _n_) the features would be (_m_ x _n_-1) and classify would be (_m_ x _1_)
10. The features are real numbers, you will need to split based on a threshold. Consider different approaches for what this threshold might be. 

First, in the `DecisionTree.__build_tree__()` method implement the above algorithm.
Next, in `DecisionTree.classify()`, write a function to produce classifications for a list of features once your decision tree has been built.


How grading works:
1. We load **_mod_complex_multi.csv_** and create our cross-validation training and test set with a `k=10` folds.  We use our own `generate_k_folds()` method.
2. We fit the (folded) training data onto the tree then classify the (folded) testing data with the tree.
3. We check the accuracy of your results versus the true results and we return the average of this over k=10 iterations.

#### Functions to complete in the `DecisionTree` class:
1. `__build_tree__()`
2. `classify()`

---

### Part 2c: Validation
_[6 pts]_

* File to use: **_mod_complex_multi.csv_**
* Grading: average test accuracy over 10 rounds should be >= 80%

In general, reserving part of your data as a test set can lead to unpredictable performance. A serendipitous choice of your training or test split could give you a very inaccurate idea of how your classifier performs. We can overcome this limitation by using k-fold cross validation.

In `generate_k_folds()`, we'll split the dataset at random into k equal subsections. Then iterating on each of our k samples, we'll reserve that sample for testing and use the other k-1 for training. Averaging the results of each fold should give us a more consistent idea of how the classifier is doing across the data as a whole.
For those who are not familiar with k folds cross-validation, please refer the tutorial here: [A Gentle Introduction to k-fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/).

How grading works:
1. The same as 2b however, we use your `generate_k_folds()` instead of ours.

#### Functions to complete in the `submission` module:
1. `generate_k_folds()`

---

### Part 3: Random Forests
_[20 pts]_

* File to use: **_mod_complex_binary.csv, mod_complex_multi.csv_**
* Grading: average test accuracy over 10 rounds should be >= 90%

The decision boundaries drawn by decision trees are very sharp, and fitting a decision tree of unbounded depth to a list of training examples almost inevitably leads to overfitting. In an attempt to decrease the variance of our classifier we're going to use a technique called 'Bootstrap Aggregating' (often abbreviated as 'bagging').

A Random Forest is a collection of decision trees, built as follows:
1. For every tree we're going to build:
   1. Subsample the examples provided us (with replacement) in accordance with a provided example subsampling rate.
   2. From the sample in the first step, choose attributes at random to learn on (in accordance with a provided attribute subsampling rate). (Without replacement)
   3. Fit a decision tree to the subsample of data we've chosen (to a certain depth).

Classification for a random forest is then done by taking a majority vote of the classifications yielded by each tree in the forest after it classifies an example.

Fill in `RandomForest.fit()` to fit the decision tree as we describe above, and fill in `RandomForest.classify()` to classify a given list of examples.

Your features and classify should be in numpy arrays where if the dataset is (_m_ x _n_) then the features is (_m_ x _n_-1) and classify is (_n_ x _1_).

To test, we will be using a forest with 5 trees, with a depth limit of 5, example subsample rate of 0.5 and attribute subsample rate of 0.5

How grading works:
1. Similar to 2b but with the call to Random Forest.

#### Functions to complete in the `RandomForest` class:
1. `fit()`
2. `classify()`

---

### Part 4: Challenge Classifier
_[10 pts]_

* File to use: **_complex_multi.csv_**
* Grading: average training accuracy over 10 runs should be >= 90% and average testing accuracy over 10 runs should be >= 90%

The Challenge Classifier should be implemented using some sort of a **tree structure**.  Students in the past have been able to call their `RandomForest` with different parameters. We also encourage things like boosting.

You've been provided with a sample of data from a research dataset in **_challenge_train.csv_** while we have reserved a part of the dataset for testing called **_challenge_test.csv_** (which you do not have access to).

To get full points for this part of the assignment, you'll need to get at least an average accuracy of 80% on the training data you have (**_challenge_train.csv_**), and at least an average accuracy of 70% on the holdout/test set (**_challenge_test.csv_**).

#### Functions to complete in the `ChallengeClassifier` class:
1. `__init__()`
2. `fit()`
3. `classify()`

---
### Part 5: Return Your name!
_[1 pts]_
Return your name from the function `return_your_name()`

---

### Helper Notebook

#### Note: You do not need to implement anything in this notebook. This part is not graded, so you can skip this part if you wish to. This notebook is just for your understanding purpose. It will help you visualize Decision trees on the dataset provided to you.
The notebook Visualize_tree.iypnb can be use to visualize tree on the datasets. You can play with this playbook. Things you can Observe:
1. How the values are splitted?
2. What is the ginni value at leaf nodes?
3. What does internal nodes represents in this DT?
4. Why all leaf nodes are not at same depth?

Feel free to change and experiment with this notebook. you can look and use Information gain as well instead of ginni to see how the DT built based on that.

### Bonus

**Note:** This part will be ever changing. Official announcements for this bonus will be made through Piazza.

We will be having a competition using your challenge classifier and a dataset of our choice. We will provide you with a portion of the dataset as well as the testing data (but without the labels) and you will upload your solution as a csv to Kaggle. Kaggle will evaluate your scores and the classifier with the highest score will win the competition. Any ties will be broken by the submission time.

We are still figuring out all the details for this bonus. Expect the competition to be open for at least a week after the assignment ends.

##### Bonus Points Metric
All bonus points will be added to your assignment 4 grades.

1. First place:  15 bonus points
2. Second place:    12 bonus points
3. Third place:     10 bonus points
4. Everyone else who achieves an accuracy of more than a
   threshold to be determined when the challenge is finalized: 7 bonus points
