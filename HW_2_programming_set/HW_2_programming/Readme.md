# Homework 2

## Submission instructions

* Due date and time: March 2nd (Sunday) 2025, 23:59 ET

* Carmen submission: 
Submit a .zip file named `name.number.zip` (e.g., `chao.209.zip`) with the following files
  - your completed Python script `main.py`
  - your 6 generated figures `Map_background.png`, `Map_horizontal_edge.png`, `Map_vertical_edge.png`, `Map_contact_edge.png`, `3D_Y.png`, `3D_Z.png`
  - your 6 generated files `Results_Map_background.npz`, `Results_Map_horizontal_edge.npz`, `Results_Map_vertical_edge.npz`, `Results_Map_contact_edge.npz`, `Results_3D_Y.npz`, `Results_3D_Z.npz`.

* Collaboration: You may discuss the homework with your classmates. However, you must write your solutions, complete your .py files, and submit them yourself. Collaboration in the sense that each of you completes some parts and then exchanges the solutions is NOT allowed. I do expect that your solutions won't be exactly the same. In your submission, you must list with whom you have discussed the homework. Please list each classmate's name and name.number (e.g., Wei-Lun Chao, chao.209) as a row at the end of `main.py`. That is, if you discussed your homework with two classmates, your .py file will have two rows at the end. Please consult the syllabus for what is and is not acceptable collaboration.

## Implementation instructions

* Download or clone this repository.

*  You will see a PPT and PDF named `HW2`, which provides useful information for the homework assignment.

* You will see one Python script: `main.py`.

* You will see a folder `for_display`, which contains some images used for display here.

* You will see a folder `data`, which contains some images used in the homework.

* You will see a folder `result`, which will save the generated results.

* Please use python3 and write your solutions from scratch. (You must use python3.)

* **Caution! Python and NumPy's indices start from 0. That is, to get the first element in a vector, the index is 0 rather than 1.**

* We note that the provided commands are designed to work with Mac/Linux with Python version 3. If you use Windows (like me!), we recommend that you run the code in the Windows command line (CMD). You may use `py -3` instead of `python3` to run the code. You may use editors like PyCharm to write your code.

* Caution! Please do not import packages (like scikit learn) that are not listed in the provided code. In this homework, you are not allowed to use numPy or other Python libraries' built-in convolution, DFT, IDFT, and filter functions. If you use them, you will get 0 points for the entire homework. 

* Caution! Follow the instructions in each question strictly to code up your solutions. Do not change the output format. Do not modify the code unless we instruct you to do so. (You are free to play with the code but your submitted code should not contain those changes that we do not ask you to do.) A homework solution that does not match the provided setup, such as format, name, initializations, etc., will not be graded. It is your responsibility to make sure that your code runs with the provided commands and scripts.

## Installation instructions

* You will be using [NumPy] (https://numpy.org/), and your code will display your results with [matplotlib] (https://matplotlib.org/). If your computer does not have them, you may install them with the following commands:
  - for NumPy: <br/>
    do `sudo apt install python3-pip` or `pip3 install numpy`. If you are using the Windows command line, you may try `setx PATH "%PATH%;C:\Python34\Scripts"`, followed by `py -3 -mpip install numpy`.

  - for matplotlib: <br/>
    do `python3 -m pip install -U pip` and then `python3 -m pip install -U matplotlib`. If you are using the Windows command line, you may try `py -3 -mpip install -U pip` and then `py -3 -mpip install -U matplotlib`.



# Introduction

In this homework, you will implement convolution, discrete Fourier transform (DFT), some filters (convolutional kernels), and some image processing steps introduced in Lectures 10 - 13 (textbook, chapters 15 - 18). Specifically, your code will output several images or frequency responses.

* You are given several images in the `data` folder as well as the following two toy images. All of them have three color channels (red, green, and blue). The pixel values are between 0.0 to 1.0.

Cosine: ![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_2_programming_set/HW_2_programming/for_display/cosine.png)

Recntangle: ![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_2_programming_set/HW_2_programming/for_display/rectangle.png)

* Your goal is to perform convolution, DFT, and several other image processing operations on them. For example, the convoluted rectangle with an average (box) filter is as below:

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_2_programming_set/HW_2_programming/for_display/Convolution_output_rectangle_average.png)

The amplitude of the DFT output of the cosine image is as below:

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_2_programming_set/HW_2_programming/for_display/DFT_amplitude_cosine.png)



# Question 0: Get ready 

* Please overview `main.py`. It contains multiple sub-functions. Specifically, you may want to take a look at `data_loader`, `load_kernel`, `Convolution`, `Modulation`, `DFT`, and `IDFT`.

* We note that a matrix and an image have different axis ordering and direction. In numPy, for a matrix `I`,  `I[i, j]` means the i-th row (top-down) and j-th column (left-right). In this homework, however, **please treat `I` and other matrices directly as images. That is, given `I`,  `I[i, j, :]` means the R, G, B pixel values at the horizontal index i (left-right) and vertical index j (bottom-up). Namely, the color at the `(i, j)` pixel location.** Please note that i and j both start from 0.



# Question 1:  (10 pts)

* Go to the `main` function and `find if int(args.current_step) == 1:`

* Given the input image `I`, you need to perform convolution of it using `kernel`. 

* We have implemented several kernels. Your job is to complete the implementation of the `Convolution` function.
  
* You are asked to complete the function `def find_background(args, I)`, which generate the background map. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `Map_background`: a value of 1 means the corresponding pixel belongs to the backgrounds. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 1`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 1 --display --save`<br/>
This command will run your code. You will see the resulting `Map_background` displayed in your command line. 

* The code will generate `Map_background.png` and  `Results_Map_background.npz`, which you will include in your submission.

* Your result should look like:

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_1_programming_set/HW_1_programming/for_display/Map_background.png)



# Question 2 & 3:  (10 + 20 pts)

* Given the input image `I` and the edge map  `Map_edge`, you are asked to complete the function `def find_edge_type(args, I, Map_background, Map_edge, Map_surface)`, which generates the maps for horizontal, vertical, and contact edges.

* Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the three output numpy arrays `Map_horizontal_edge`, `Map_vertical_edge`, and `Map_contact_edge`: a value of 1 means a pixel belongs to the corresponding type of edges. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 2`** and **`#### Your job 3`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* **Hint:** Since we already provided `Map_edge`, which locates all the pixels belonging to edges, `Map_horizontal_edge[i, j]`, `Map_vertical_edge[i, j]`, and `Map_contact_edge[i, j]` can be 1 only when `Map_edge[i, j]` is 1.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 3 --display --save`<br/>
This command will run your code. You will see the resulting `Map_horizontal_edge`, `Map_vertical_edge`, and `Map_contact_edge` displayed in your command line. 

* The code will generate `Map_horizontal_edge.png`, `Map_vertical_edge.png`, `Map_contact_edge.png`, `Results_Map_horizontal_edge.npz`, `Results_Map_vertical_edge.npz`, and `Results_Map_contact_edge.npz`, which you will include in your submission.

* Your result should look like:

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_1_programming_set/HW_1_programming/for_display/Map_horizontal_edge.png)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_1_programming_set/HW_1_programming/for_display/Map_vertical_edge.png)

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_1_programming_set/HW_1_programming/for_display/Map_contact_edge.png)



# Question 4: (50 pts)

* Given the input image `I`, the background map `Map_background`, surface map `Map_surface`, edge maps for different types `Map_horizontal_edge`, `Map_vertical_edge`, `Map_contact_edge`, and the camera angle `theta`, you are asked to complete the function `def Y_3D(I, Map_background, Map_horizontal_edge, Map_vertical_edge, Map_contact_edge, Map_surface, theta)`, which generates the Y (height) map in 3D.

* Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `Y`. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 4`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* **Hint:** You need to create a matrix `A` and a vector `b` to apply the linear algebra formula to obtain the answer. 

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 4 --display --save`<br/>
This command will run your code. You will see the resulting `3D_Y` displayed in your command line. 

* The code will generate `3D_Y.png` and `Results_3D_Y.npz`, which you will include in your submission.



# Question 5: (10 pts)

* Given the input image `I`, `Y`, the camera angle `theta`, and each pixel's 2D vertical location `yy`, you are asked to complete the function `def Z_3D(I, yy, Y, theta)`, which generates the Z (depth) map in 3D.

* Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `Z`. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 5`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 5 --display --save`<br/>
This command will run your code. You will see the resulting `3D_Z` displayed in your command line. 

* The code will generate `3D_Z.png` and `Results_3D_Z.npz`, which you will include in your submission.



# What to submit:

* Please see the beginning of the page. Please follow **Submission instructions** to submit a .zip file named name.number.zip (e.g., chao.209.zip). Failing to submit a single .zip file will not be graded.
