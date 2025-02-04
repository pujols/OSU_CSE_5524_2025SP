# Homework 1

## Submission instructions

* Due date and time: February 18th (Monday) 2025, 23:59 ET

* Carmen submission: 
Submit a .zip file named `name.number.zip` (e.g., `chao.209.zip`) with the following files
  - your completed Python script `main.py`
  - your 6 generated figures `Map_background.png`, `Map_horizontal_edge.png`, `Map_vertical_edge.png`, `Map_contact_edge.png`, `3D_Y.png`, `3D_Z.png`
  - your 6 generated files `Results_Map_background.npz`, `Results_Map_horizontal_edge.npz`, `Results_Map_vertical_edge.npz`, `Results_Map_contact_edge.npz`, `Results_3D_Y.npz`, `Results_3D_Z.npz`.

* Collaboration: You may discuss the homework with your classmates. However, you need to write your solutions, complete your .py files, and submit them by yourself. In your submission, you need to list with whom you have discussed the homework. Please list each classmate’s name and name.number (e.g., Wei-Lun Chao, chao.209) as a row at the end of `main.py`. That is, if you discussed your homework with two classmates, your .py file will have two rows at the end. Please consult the syllabus for what is and is not acceptable collaboration.

## Implementation instructions

* Download or clone this repository.

* You will see two Python scripts: `main.py` and `numpy_example.py`.

* You will see a folder `for_display`, which simply contains some images used for display here.

* Please use python3 and write your solutions from scratch. (You must use python3.)

* **Caution! Python and NumPy's indices start from 0. That is, to get the first element in a vector, the index is 0 rather than 1.**

* We note that the provided commands are designed to work with Mac/Linux with Python version 3. If you use Windows (like me!), we recommend that you run the code in the Windows command line (CMD). You may use `py -3` instead of `python3` to run the code. You may use editors like PyCharm to write your code.

* Caution! Please do not import packages (like scikit learn) that are not listed in the provided code. Follow the instructions in each question strictly to code up your solutions. Do not change the output format. Do not modify the code unless we instruct you to do so. (You are free to play with the code but your submitted code should not contain those changes that we do not ask you to do.) A homework solution that does not match the provided setup, such as format, name, initializations, etc., will not be graded. It is your responsibility to make sure that your code runs with the provided commands and scripts.

## Installation instructions

* You will be using [NumPy] (https://numpy.org/), and your code will display your results with [matplotlib] (https://matplotlib.org/). If your computer does not have them, you may install them with the following commands:
  - for NumPy: <br/>
    do `sudo apt install python3-pip` or `pip3 install numpy`. If you are using the Windows command line, you may try `setx PATH "%PATH%;C:\Python34\Scripts"`, followed by `py -3 -mpip install numpy`.

  - for matplotlib: <br/>
    do `python3 -m pip install -U pip` and then `python3 -m pip install -U matplotlib`. If you are using the Windows command line, you may try `py -3 -mpip install -U pip` and then `py -3 -mpip install -U matplotlib`.



# Introduction

In this homework, you will implement a simplified version of the simple version system introduced in Lectures 2 and 3 (textbook, chapter 2). Specifically, your code will output several maps, including the 3D Y and 3D Z maps.

* Specifically, you are given the following gray-scale image I (a 2D matrix) captured by parallel projection with a viewing angle of theta = 45 degrees. White color means a pixel value of 1; Gray color means a pixel value of 0.5; black color means a pixel value of 0.0.

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_1_programming_set/HW_1_programming/for_display/I.png)

* Your goal is to derive the 3D locations of each pixel. Specifically, you are tasked to derive the 3D Y (height) map and 3D Z (depth) map for each pixel.

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_1_programming_set/HW_1_programming/for_display/3D_Y.png)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_1_programming_set/HW_1_programming/for_display/3D_Z.png)

As you can see, the Y and Z maps have the same size as the input image I. For each pixel location, the input image I records its color or light intensity, while the Y and Z maps record their height and depth in 3D.

# Question 0: Exercise

* You will use [NumPy] (https://numpy.org/) extensively in this homework. NumPy is a library for the Python programming language, which adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. NumPy has many great functions and operations that will make your implementation much easier. 

* If you are not familiar with Numpy, we recommend that you read this [tutorial] (https://cs231n.github.io/python-numpy-tutorial/) or some other tutorials online, and then play with some code to get familiar with it.

* We have provided some useful Numpy operations that you may want to use in `numpy_example.py`. You may want to comment out all the lines first and execute them one by one or in a group to see the results and the differences. You can run the command `python3 numpy_example.py`.

* Caution! Python and NumPy's indices start from 0. That is, to get the first element in a vector, the index is 0 rather than 1.



# Question 1: Linear and nonlinear regression (30 pts)

* You will implement linear and nonlinear regression (with the RSS loss) in this question. You are to amend your implementation into `Regression.py`.

* There are many sub-functions in `Regression.py`. You can ignore all of them but `def linear_regression(X, Y)` and `def main(args)`. In `main(args)`, you will see a general pipeline of machine learning: <br/>
  - Loading data: `X_original, Y_original = data_loader(args)`, in which `X_original` is a 1-by-N matrix (numpy array) and each column is a data instance. You can type `X[:, 0]` to extract the "first" data instance from `X_original`. (Caution! python and numpy's indices start from 0. That is, to get the first element in a vector, the index is 0 rather than 1.) <br/>
  - Data separation: Separate data into training, validation, and test sets.
  - Training_and_validation: the for loop `for poly in range(1, 12)` will try different polynomial feature transforms and check which one leads to the smallest validation error.
  - Feature transform: `X_train = polynomial_transform(np.copy(X_original_train), int(poly))` extends each column of `X` to its polynomial representation. For example, given x, this transform will extends it to [x, x^2, ..., x^(int(poly))]^T.
  - Learning patterns: `w, b = linear_regression(X_train, Y_train)`, in which the code takes `X_train` and the desired labels `Y_train` as input and output the weights `w` and the bias `b`.
  - Apply the learned patterns to the data: `train_error = np.mean((np.matmul(w.transpose(), X_train) + b - Y_train.transpose()) ** 2)` and `test_error = np.mean((np.matmul(w.transpose(), X_test) + b - Y_test.transpose()) ** 2)` compute the training and test error.
  
## Coding (15/30 pts):

You have two parts to implement in `Regression.py`:

* The function `def linear_regression(X, Y)`: please go to the function and read the input format, output format, and the instructions carefully. You can assume that the actual inputs will follow the input format, and your goal is to generate the required numpy arrays (`w` and `b`), i.e., the weights and the bias of linear regression. Please make sure that your results follow the required numpy array shapes. You are to implement your code between `### Your job 1 starts here ###` and `### Your job 1 ends here ###`. Please note that `1` has not been appended to `X`. 1. If you want to apply the closed-form solution directly, you have to append `1` into your data `X` (for every column). You are free to create more space between those two lines: we include them just to explicitly tell you where you are going to implement them.

* The decision of which polynomial degree to use via the validation process: please go to `def main(args)`. You are to implement your code between `### Your job 2 starts here ###` and `### Your job 2 ends here ###`. You will see some instructions there. You are free to create more space between those two lines: we include them just to explicitly tell you where you are going to implement them.

## Auto grader:

* You may run the following command to test your implementation<br/>
`python3 Regression.py --data simple --auto_grade`<br/>
Note that, the auto grader is to check your implementation semantics. If you have syntax errors, you may get python error messages before you can see the auto_graders' results.

* Again, the auto_grader is just to simply check your implementation for a toy case. It will not be used for your final grading.

## Play with different datasets (Task 1 - linear testing, 2/30 pts):

* Please run the following command<br/>
`python3 Regression.py --data linear --polynomial 1 --display --save`<br/>
This command will run linear regression on 1D linear data (the x-axis is the feature and the y-axis is the label). You will see the resulting `w` and `b` being displayed in your command line. You will also see the training (on red points) and test error (on blue points). 

* The code will generate `linear_1.png` and `Results_linear_1.npz`, which you will include in your submission.

* You may play with other commands by (1) removing `--save` (2) changing the `--polynomial 1` to a non-negative integer (e.g, 2, 3, ..., 12). You will see that, while larger values lead to smaller training errors, the test error is not necessarily lower. For a very large value, the test error can go very large.

## Play with different datasets (Task 2 - quadratic data, 2/30 pts):

* Please run the following command<br/>
`python3 Regression.py --data quadratic --polynomial 2 --display --save`<br/>
This command will run linear regression on 1D quadratic data (the x-axis is the feature and the y-axis is the label). The code will produce a polynomial = 2 representation for the data (i.e., `X` becomes 2-by-N). You will see the resulting `w` and `b` being displayed in your command line. You will also see the training (on red points) and test error (on blue points). 

* The code will generate `quadratic_2.png` and `Results_quadratic_2.npz`, which you will include in your submission.

* You may play with other commands by (1) removing `--save` (2) changing the `--polynomial 2` to a non-negative integer (e.g, 1, 3, ..., 12). You will see that, while larger values lead to smaller training errors, the test error is not necessarily lower. For a very large value, the test error can go very large.

## Play with different datasets (Task 3 - unknown degree data, 2/30 pts):

* Please run the following command<br/>
`python3 Regression.py --data unknown --polynomial 5 --display --save`<br/>
This command will run linear regression on 1D data (the x-axis is the feature and the y-axis is the label). The code will produce a polynomial = 5 representation for the data (i.e., `X` becomes 5-by-N). You will see the resulting `w` and `b` being displayed in your command line. You will also see the training (on red points) and test error (on blue points). 

* The code will generate `unknown_5.png` and `Results_unknown_5.npz`, which you will include in your submission.

## Play with different datasets (Task 4 - unknown degree noisy data, 2/30 pts):

* Please run the following command<br/>
`python3 Regression.py --data unknown_noise --polynomial 5 --display --save`<br/>
This command will run linear regression on 1D data (the x-axis is the feature and the y-axis is the label), which is exactly the same as in Task 3 but with additional noise. The code will produce a polynomial = 5 representation for the data (i.e., `X` becomes 5-by-N). You will see the resulting `w` and `b` being displayed in your command line. You will also see the training (on red points) and test error (on blue points). 

* The code will generate `unknown_noise_5.png` and `Results_unknown_noise_5.npz`, which you will include in your submission.

## Play with training_validation_testing (Task 5, 7/30 pts):

* Please run the following commands<br/>
`python3 Regression.py --data linear --validation --display`<br/>
`python3 Regression.py --data quadratic --validation --display`<br/>
`python3 Regression.py --data unknown --validation --display`<br/>
`python3 Regression.py --data unknown_noise --validation --display`<br/>
These commands will select for each data the corresponding best polynomial degree, based on the validation error. Please write down for each data the best polynomial degree and the corresponding training and test error. Please discuss for `unknown_noise`, why its best polynomial degree is different from that of `unknown`.





# Question 2: KNN for classification (20 pts)

* You will implement KNN for "binary" classification in this question. The data are images of digits 1 and 8. You are to amend your implementation into `KNN.py`.

* There are many sub-functions in `KNN.py`. You can ignore all of them but `def distance(x_test, x_train, dis_metric)`, `def KNN(x_test, X, Y, K, dis_metric)`, and `def main(args)`. In `main(args)`, you are to implement the leave-one-out cross-validation to find the `best_dis_metric` from ["L1", "L2", "cosine"], the `best_K` from [1, 3, 5, 7, 9, 11, 13, 15, 17], and the corresponding `best_val_accuracy` and `test_accuracy`.<br/>

* The accuracy is "average 1[y_n = predict_y_n]". That is, when an example is correctly classified, you get 1; otherwise, 0. You then average this over all data.

The data X is in the column form: i.e., a column is a data instance.
  
## Coding (15/20 pts):

You have three parts to implement in `KNN.py`:

* The function `def distance(x_test, x_train, dis_metric)`: please go to the function and read the input format, output format, and the instructions carefully. You can assume that the actual inputs will follow the input format, and your goal is to complete the distance computation for "L1", "L2", and "cosine" distance. You are to implement your code between `### Your job 1 starts here ###` and `### Your job 1 ends here ###`. We have provided the code skeleton. You are free to create more space between those two lines: we include them just to explicitly tell you where you are going to implement.

* The function `KNN(x_test, X, Y, K, dis_metric)`: please go to the function and read the input format, output format, and the instructions carefully. You can assume that the actual inputs will follow the input format, and your goal is to complete the KNN rule, using the `def distance(x_test, x_train, dis_metric)` function you have implemented which computes the distance between two data instances. The output `y_predict` will be the predicted label (either -1 or 1). You are to implement your code between `### Your job 2 starts here ###` and `### Your job 2 ends here ###`. You are free to create more space between those two lines: we include them just to explicitly tell you where you are going to implement.

* The leave-one-out cross-validation in `main(args)`: You are to implement your code between `### Your job 3 starts here ###` and `### Your job 3 ends here ###`. After this part of the code, you should output the `best_dis_metric` from ["L1", "L2", "cosine"], the `best_K` from [1, 3, 5, 7, 9, 11, 13, 15, 17], and the corresponding `best_val_accuracy` and `test_accuracy`. Note that, the accuracy should be between 0 and 1. You are free to create more space between those two lines: we include them just to explicitly tell you where you are going to implement.

## Play with the data (5/20 pts):

* Please run the following command<br/>
`python3 KNN.py`<br/>

Please report `best_dis_metric`, `best_K`, `best_val_accuracy`, and `test_accuracy` in the PDF.





# What to submit:

* Please see the beginning of the page. Please follow **Submission instructions** to submit a .zip file named name.number.zip (e.g., chao.209.zip). Failing to submit a single .zip file will not be graded.





# What to report in `name.number.pdf`

* For Question 1, Task 5, please write down for each data the best polynomial degree and the corresponding training and test error. Please discuss for `unknown_noise`, why its best polynomial degree is different from that of `unknown`.

* For Question 2, please report `best_dis_metric`, `best_K`, `best_val_accuracy` , and `test_accuracy` in the PDF.
