Download Link: https://assignmentchef.com/product/solved-cs-189-introduction-to-machine-learning-hw1
<br>
This homework is comprised of two parts. The first part consists of a set of coding exercises. The second part consists of math problems.

Start this homework early! You can only submit to Kaggle twice a day.

Deliverables:

<ol>

 <li>Submit your predictions for the test sets to Kaggle as early as possible. Include your Kaggle scores in your write-up (see below).</li>

 <li>Submit a PDF of your homework, with an appendix listing all your code, to the Gradescope assignment entitled “HW1 Write-Up”. You may typeset your homework in LaTeX or Word (submit PDF format, not .doc/.docx format) or submit neatly handwritten and scanned solutions. Please start each question on a new page. If there are graphs, include those graphs in the correct sections. Do not put them in an appendix. We need each solution to be self-contained on pages of its own.</li>

</ol>

<ul>

 <li>In your write-up, please state with whom you worked on the homework. • In your write-up, please copy the following statement and sign your signature next to it. (Mac Preview and FoxIt PDF Reader, among others, have tools to let you sign a PDF file.) We want to make it <em>extra </em>clear so that no one inadverdently cheats.</li>

</ul>

<em>“I certify that all solutions are entirely in my own words and that I have not looked at another student’s solutions. I have given credit to all external sources I consulted.”</em>

<ol start="3">

 <li>Submit all the code needed to reproduce your results to the Gradescope assignment entitled “HW1 Code”. Yes, you must submit your code twice: once in your PDF write-up (above) so the readers can easily read it, and once in compilable/interpretable form so the readers can easily run it. Do NOT include any data files we provided. Please include a short file named README listing your name, student ID, and instructions on how to reproduce your results. Please take care that your code doesn’t take up inordinate amounts of time or memory. If your code cannot be executed, your solution cannot be verified.</li>

</ol>

<h1>1         Python Configuration and Data Loading</h1>

Please follow the instructions below to ensure your Python environment is configured properly, and you are able to successfully load the data provided with this homework. No solution needs to be submitted for this question. For all coding questions, we recommend using <a href="https://store.continuum.io/cshop/anaconda">Anaconda</a> for Python 3.

<ul>

 <li>Either install Anaconda for Python 3, or ensure you’re using Python 3. To ensure you’re running Python 3, open a terminal in your operating system and execute the following command:</li>

</ul>

python –version

Do not proceed until you’re running Python 3.

<ul>

 <li>Install the following dependencies required for this homework by executing the following command in your operating system’s terminal:</li>

</ul>

pip install scikit-learn scipy numpy matplotlib

Please use Python 3 with the modules specified above to complete this homework.

<ul>

 <li>You will be running out-of-the-box implementations of support vector machines to classify three datasets. You will find a set of .mat files in the data folder for this homework. Each .mat file will load as a Python dictionary. Each dictionary contains three fields:

  <ul>

   <li>training data, the training set features. Rows are samples and columns are features.</li>

   <li>training labels, the training set labels. Rows are samples. There is one column: The label for each sample.</li>

   <li>test data, the test set features. Rows are samples and columns are features. You will fit a model to predict the labels for this test set, and submit those predictions to Kaggle.</li>

  </ul></li>

</ul>

The three datasets for the coding portion of this assignment are described below.

<ul>

 <li>mnist data.mat contains data from the MNIST dataset. There are 60,000 labeled digit images for training, and 10,000 digit images for testing. The images are grayscale, 28×28 pixels flattened. There are 10 possible labels for each image, namely, the digits 0–9.</li>

</ul>

Figure 1: Examples from the MNIST dataset.

<ul>

 <li>spam data.mat contains featurized spam data. The labels are 1 for spam and 0 for ham. The data folder includes the script featurize.py and the folders spam, ham (not spam), and test (unlabeled test data); you may modify featurize.py to generate new features for the spam data.</li>

 <li>cifar10 data.mat contains data from the CIFAR10 dataset. There are 50,000 labeled object images for training, and 10,000 object images for testing. The images are flattened 3×32×32 (3 color channels). The labels 0–9 correspond alphabetically to the categories.</li>

</ul>

For example, 0 means airplane, 1 means automobile, 2 means bird, and so on.

Figure 2: Examples from the CIFAR-10 dataset.

To check whether your Python environment is configured properly for this homework, ensure the following Python script executes without error. Pay attention to errors raised when attempting to import any dependencies. Resolve such errors by manually installing the required dependency (e.g. execute pip install numpy for import errors relating to the numpy package).

<table width="607">

 <tbody>

  <tr>

   <td width="607">import sys if sys.version_info[0] &lt; 3:raise Exception(“Python 3 not detected.”)import numpy as np import matplotlib.pyplot as plt from sklearn import svm from scipy import iofor data_name in [“mnist”, “spam”, “cifar10”]: data = io.loadmat(“data/%s_data.mat” % data_name) print(“
loaded %s data!” % data_name) fields = “test_data”, “training_data”, “training_labels” for field in fields:print(field, data[field].shape)</td>

  </tr>

 </tbody>

</table>

<h1>2       Data Partitioning</h1>

Rarely will you receive “training” data and “validation” data; usually you will have to partition available labeled data yourself. The datasets for this assignment are described below. Write code to partition the datasets as follows.

<ul>

 <li>For the MNIST dataset, write code that sets aside 10,000 training images as a validation set.</li>

 <li>For the spam dataset, write code that sets aside 20% of the training data as a validation set.</li>

 <li>For the CIFAR-10 dataset, write code that sets aside 5,000 training images as a validation set. Be sure to shuffle your data before splitting it to make sure all the classes are represented in your partitions.</li>

</ul>

<h1>3        Support Vector Machines: Coding</h1>

We will use linear support vector machines to classify our datasets. For images, we will use the simplest of features for classification: raw pixel brightness values. In other words, our feature vector for an image will be a row vector with all the pixel values concatenated in a row major (or column major) order.

There are several ways to evaluate models. We will use <em>classification accuracy </em>as a measure of the error rate (see here: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html">https://scikit-learn.org/stable/modules/generated/sklearn. </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html">metrics.accuracy_score.html</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html">)</a>.

Train a linear support vector machine (SVM) on all three datasets. Plot the error rate on the training and validation sets versus the number of training examples that you used to train your classifier. The number of training examples in your experiment will vary per dataset.

You may only use sklearn for the SVM model and the accuracy metric function. Everything else (train vs. val plots) must be done without the use of sklearn.

<ul>

 <li>For the MNIST dataset, use raw pixels as features. Train your model with the following numbers of training examples: 100, 200, 500, 1,000, 2,000, 5,000, 10,000. At this stage, you should expect accuracies between 70% and 90%.</li>

</ul>

Hint: Be consistent with any preprocessing you do. Use either integer values between 0 and 255 or floating-point values between 0 and 1. Training on floats and then testing with integers is bound to cause trouble.

<ul>

 <li>For the spam dataset, use the provided word frequencies as features. In other words, each document is represented by a vector, where the <em>i</em>th entry denotes the number of times word <em>i </em>(as specified in featurize.py) is found in that document. Train your model with the following numbers of training examples: 100, 200, 500, 1,000, 2,000, ALL.</li>

</ul>

Note that this dataset does not have 10,000 examples; use all of your examples instead of 10,000. At this stage, you should expect accuracies between 70% and 90%.

<ul>

 <li>For the CIFAR-10 dataset, use raw pixels as features. At this stage, you should expect accuracies between 25% and 35%. Be forwarned that training SVMs for CIFAR-10 takes a couple minutes to run for a large training set. Train your model with the following numbers of training examples: 100, 200, 500, 1,000, 2,000, 5,000.</li>

</ul>

Note: We find that SVC(kernel=’linear’) is faster than LinearSVC.

<h1>4       Hyperparameter Tuning</h1>

In the previous problem, you learned parameters for a model that classifies the data. Many classifiers also have <em>hyperparameters </em>that you can tune that influence the parameters. In this problem, we’ll determine good values for the regularization parameter <em>C </em>in the soft-margin SVM algorithm.

When we are trying to choose a hyperparameter value, we train the model repeatedly with different hyperparameters. We select the hyperparameter that gives the model with the highest accuracy on the validation dataset. Before generating predictions for the test set, the model should be retrained using all the labeled data (including the validation data) and the previously-determined hyperparameter.

The use of automatic hyperparameter optimization libraries is prohibited for this part of the homework.

(a) For the MNIST dataset, find the best <em>C </em>value. In your report, list the <em>C </em>values you tried, the corresponding accuracies, and the best <em>C </em>value. As in the previous problem, for performance reasons, you are required to train with up to 10,000 training examples but not required to train with more than that.

<h1>5        K-Fold Cross-Validation</h1>

For smaller datasets (e.g., the spam dataset), the validation set contains fewer examples, and our estimate of our error might not be accurate—the estimate has high variance. A way to combat this is to use <em>k-fold cross-validation</em>.

In <em>k</em>-fold cross-validation, the training data is shuffled and partitioned into <em>k </em>disjoint sets. Then the model is trained on <em>k </em>− 1 sets and validated on the <em>k<sup>th </sup></em>set. This process is repeated <em>k </em>times with each set chosen as the validation set once. The cross-validation accuracy we report is the accuracy averaged over the <em>k </em>iterations.

Use of automatic cross-validation libraries is prohibited for this part of the homework.

(a) For the spam dataset, use 5-fold cross-validation to find and report the best <em>C </em>value. In your report, list the <em>C </em>values you tried, the corresponding accuracies, and the best <em>C </em>value.

Hint: Effective cross-validation requires choosing from random partitions. This is best implemented by randomly shuffling your training examples and labels, then partitioning them by their indices.

<h1>6      Kaggle</h1>

<ul>

 <li>MNIST Competition: <a href="https://www.kaggle.com/c/cs189-hw1-mnist">https:</a><a href="https://www.kaggle.com/c/cs189-hw1-mnist">//</a><a href="https://www.kaggle.com/c/cs189-hw1-mnist">kaggle.com</a><a href="https://www.kaggle.com/c/cs189-hw1-mnist">/</a><a href="https://www.kaggle.com/c/cs189-hw1-mnist">c</a><a href="https://www.kaggle.com/c/cs189-hw1-mnist">/</a><a href="https://www.kaggle.com/c/cs189-hw1-mnist">cs189-hw1-mnist</a></li>

 <li>SPAM Competition: <a href="https://www.kaggle.com/c/cs189-hw1-spam">https:</a><a href="https://www.kaggle.com/c/cs189-hw1-spam">//</a><a href="https://www.kaggle.com/c/cs189-hw1-spam">kaggle.com</a><a href="https://www.kaggle.com/c/cs189-hw1-spam">/</a><a href="https://www.kaggle.com/c/cs189-hw1-spam">c</a><a href="https://www.kaggle.com/c/cs189-hw1-spam">/</a><a href="https://www.kaggle.com/c/cs189-hw1-spam">cs189-hw1-spam</a></li>

 <li>CIFAR-10 Competition: <a href="https://www.kaggle.com/c/cs189-hw1-cifar10">https:</a><a href="https://www.kaggle.com/c/cs189-hw1-cifar10">//</a><a href="https://www.kaggle.com/c/cs189-hw1-cifar10">kaggle.com</a><a href="https://www.kaggle.com/c/cs189-hw1-cifar10">/</a><a href="https://www.kaggle.com/c/cs189-hw1-cifar10">c</a><a href="https://www.kaggle.com/c/cs189-hw1-cifar10">/</a><a href="https://www.kaggle.com/c/cs189-hw1-cifar10">cs189-hw1-cifar10</a></li>

</ul>

Using the best model you trained for each dataset, generate predictions for the test sets we provide and save those predictions to .csv files. Be sure to use integer labels (not floating-point!) and no spaces (not even after the commas). Upload your predictions to the Kaggle leaderboards (submission instructions are provided within each Kaggle competition). In your report, include your Kaggle name as it displays on the leaderboard and your Kaggle score for each of the three datasets.

For your Kaggle submissions, you may optionally add more features or use a non-linear SVM kernel to get a higher position on the leaderboard. If you do, please explain what you did in your report and cite your external sources. Examples of things you might investigate include SIFT and HOG features for images, and bag of words for spam/ham. Almost everything is fair game as long as your underlying model is an SVM (i.e., do not use a neural network, decision tree, etc.). You are also not allowed to search for the labeled test data and submit that to Kaggle. If you have any questions about whether something is allowed or not, ask on Piazza.

Remember to start early! Kaggle only permits two submissions per leaderboard per day. To help you format the submission, please use check.py to run a basic sanity check on your submission and save csv.py to help save your results. To check your submission csv,

python check.py &lt;competition name, eg. mnist&gt; &lt;submission csv file&gt;

<h1>7         Theory of Hard-Margin Support Vector Machines</h1>

A <em>decision rule </em>(or <em>classifier</em>) is a function <em>r </em>: R<em><sup>d </sup></em>→ ±1 that maps a feature vector (test point) to

+1 (“in class”) or −1 (“not in class”). The decision rule for linear SVMs is



<em>r</em>(<em>x</em>) = <sub> </sub>+1    if <em>w </em>· <em>x </em>+ α ≥ 0,                                                    (1)

<sup></sup><sub> </sub>−1  otherwise,

where <em>w </em>∈ R<em><sup>n </sup></em>and α ∈ R are the weights (parameters) of the SVM. The hard-margin SVM optimization problem (which chooses the weights) is

min |<em>w</em>|<sup>2 </sup>subject to <em>y<sub>i</sub></em>(<em>X<sub>i </sub></em>· <em>w </em>+ α) ≥ 1, ∀<em>i </em>∈ {1,…,<em>m</em>},      (2) <em>w</em>,α

√

where |<em>w</em>| = k<em>w</em>k<sub>2 </sub>=   <em>w </em>· <em>w</em>.

We can rewrite this optimization problem by using Lagrange multipliers to eliminate the constraints. (If you’re curious to know what Lagrange multipliers are, the Wikipedia page is recommended, but you don’t need to understand them to do this problem.) We thereby obtain the equivalent optimization problem

<em>m</em>

<sup>X</sup>

<table width="623">

 <tbody>

  <tr>

   <td width="521">maxmin |<em>w</em>|<sup>2 </sup>− λ<em><sub>i</sub></em>(<em>y<sub>i</sub></em>(<em>X<sub>i </sub></em>· <em>w </em>+ α) − 1). λ<em><sub>i</sub></em>≥0 <em>w</em>,α <em>i</em>=1(a) Show that Equation (3) can be rewritten as the <em>dual optimization problem</em><em>                                                         m                   m     m</em></td>

   <td width="102">(3)</td>

  </tr>

 </tbody>

</table>

<h2>                                                                      XXX                                                       X</h2>

maxλ<em><sub>i</sub></em>λ<em><sub>j</sub>y<sub>i</sub>y<sub>j</sub>X<sub>i </sub></em>· <em>X<sub>j </sub></em>subject to           λ<em><sub>i</sub>y<sub>i </sub></em>= 0. (4) λ<em><sub>i</sub></em>≥0

<em>i</em>=1<em>i</em>=1 <em>j</em>=1                                                                      <em>i</em>=1

Hint: Use calculus to determine what values of <em>w </em>and α optimize Equation (3). Explain where the new constraint comes from.

We note that SVM software usually solves this dual quadratic program, not the primal quadratic program.

<ul>

 <li>Suppose we know the values λ<sup>∗</sup><em><sub>i </sub></em>and α<sup>∗ </sup>that optimize Equation (3). Show that the decision rule specified by Equation (1) can be written</li>

</ul>



<em>r</em>(<em>x</em>) =  +1   if <em> y<sub>i</sub>X<sub>i </sub></em>· <em>x </em>≥ 0,                                       (5)

 −1  otherwise.

<ul>

 <li>The training points <em>X<sub>i </sub></em>for which λ<sup>∗</sup><em><sub>i </sub></em>&gt; 0 are called the support vectors. In practice, we frequently encounter training data sets for which the support vectors are a small minority of the training points, especially when the number of training points is much larger than the number of features (i.e., the dimension of the feature space). Explain why the support vectors are the only training points needed to evaluate the decision rule. Then explain why the non-support vectors nonetheless still have some influence on the decision rule …what is the nature of that influence?</li>

</ul>