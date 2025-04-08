# Homework 2

## Submission instructions

* Due date and time: March 2nd (Sunday) 2025, 23:59 ET

* Carmen submission: 
Submit a .zip file named `name.number.zip` (e.g., `chao.209.zip`) with the following files
  - your completed Python script `main.py`
  - your generated figures in the .png format (please see each question for what needs to be submitted) 
  - your generated files in the .npz format (please see each question for what needs to be submitted)
  - a single-page PDF named `report.pdf` (font no smaller than 12) describing your solution to questions 6 and 8.

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

* **Caution! Please do not import packages (like scikit learn) that are not listed in the provided code. In this homework, you are not allowed to use numPy's or other Python libraries' built-in convolution, DFT, IDFT, and filter functions. If you use them, you will get 0 points for the entire homework.** 

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

The amplitude of the DFT output of the cosine image is as below (think about why there is a point in the middle at frequency = 0, 0):

![Alt text](https://github.com/pujols/OSU_CSE_5524_2025SP/blob/main/HW_2_programming_set/HW_2_programming/for_display/DFT_amplitude_cosine.png)



# Question 0: Get ready 

* Please overview `main.py`. It contains multiple sub-functions. Specifically, you may want to take a look at `data_loader`, `load_kernel`, `Convolution`, `Modulation`, `DFT`, and `IDFT`.

* We note that a matrix and an image have different axis ordering and direction. In numPy, for a matrix `I`,  `I[i, j]` means the i-th row (top-down) and j-th column (left-right). In this homework, however, **please treat `I` and other matrices directly as images. That is, given `I`,  `I[i, j, :]` means the R, G, and B pixel values at the horizontal index i (left-right) and vertical index j (bottom-up). Namely, the color at the `(i, j)` pixel location.** Please note that i and j both start from 0.



# Question 1:  (20 pts)

* Go to the `main` function and find `if int(args.current_step) == 1:`

* Given the input image `I`, you need to perform convolution of it using `kernel`. 

* We have implemented several kernels in the `load_kernel(args)` function.

* Your job is to complete the implementation of the `Convolution(args, I, kernel)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy array `I_out`. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 1.0`** and **`#### Your job 1.1`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Caution! For this question, please follow the formula in `HW2.ppt` or `HW2.pdf`.

## Running and saving

* Once completed, please run the following commands<br/>
`python3 main.py --current_step 1 --data rectangle --kernel average --display --save`<br/>
`python3 main.py --current_step 1 --data dreese --kernel move_avg --display --save`<br/>
These commands will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `1_Convolution_output_rectangle_average.png`, `1_Convolution_output_dreese_move_avg.png`, `1_Results_Convolution_output_dreese_move_avg.npz`, and `1_Results_Convolution_output_rectangle_average.npz`, which you will include in your submission.


# Question 2:  (30 pts)

* Go to the `main` function and find `if int(args.current_step) == 2:` and read the corresponding code.

* Given the input image `I`, you need to perform DFT and then IDFT on it.

* We have implemented the `IDFT(args, I_amplitude, I_phase)` function.

* Your job is to complete the implementation of the `DFT(args, I)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format, and your goal is to generate the output numpy arrays `I_out_real` and `I_out_imaginary`, corresponding to the real and imaginary components of the DFT frequency responses. (We have implemented the part for how to convert them into `I_out_amplitude` and `I_out_phase`.) Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 2`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Caution! For this question, please follow the formula in `HW2.ppt` or `HW2.pdf`.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 2 --data cosine --display --save`<br/>
`python3 main.py --current_step 2 --data dreese --display --save`<br/>
These commands will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `2_IDFT_image_dreese.png`, `2_IDFT_image_cosine.png`, `2_DFT_amplitude_dreese.png`, `2_DFT_amplitude_cosine.png`, `2_Results_DFT_amplitude_dreese.npz`, `2_Results_DFT_amplitude_cosine.npz`, `2_Results_IDFT_image_dreese.npz`, and `2_Results_IDFT_image_cosine.npz`, which you will include in your submission.


# Question 3: (0 pts)

* Go to the `main` function and find `if int(args.current_step) == 3:` and read the corresponding code.

* This part of the code is about modulation, and we have implemented the `Modulation(args, I, u_freq, v_freq)` function. Please read through it.

* Your job is to understand how modulation changes an image and verify that modulation in the spatial domain is equivalent to convolution in the frequency domain. Specifically, `I_mod_1` will look very similar to `I_mod_2`.

## Running and saving

* Please run the following command<br/>
`python3 main.py --current_step 3 --data rectangle --display`<br/>
This command will run your code. You will see several generated images, and several texts displayed in command lines.

* No files need to be submitted for this question.


# Question 4: (0 pts)

* Go to the `main` function and find `if int(args.current_step) == 4:` and read the corresponding code.

* This part of the code is about the relationship between convolution in the spatial domain and multiplication in the frequency domain. Specifically, we implemented the code where the convolutional kernel can inferred from division in the frequency domain.   

* Your job is to understand and verify the relationship. Specifically, `kernel` will look very similar to `kernel_reconstruct`.
  
## Running and saving

* Please run the following command<br/>
`python3 main.py --current_step 4 --data rectangle --kernel translate_top_right --display`<br/>
This command will run your code. You will see several generated images, and several texts displayed in command lines.

* No files need to be submitted for this question.


# Question 5: (0 pts)

* Go to the `main` function and find `if int(args.current_step) == 5:` and read the corresponding code.

* This part of the code is about switching the phases between two images.

## Running and saving

* Please run the following command<br/>
`python3 main.py --current_step 5 --data dreese --display`<br/>
This command will run your code. You will see several generated images, and several texts displayed in command lines.

* No files need to be submitted for this question.
  

# Question 6: (20 pts)

* Go to the `main` function and find `if int(args.current_step) == 6:` and read the corresponding code.

* Given the input image `I` augmented with noise, i.e., `I_noisy`, you need to perform convolution to attempt to remove the noise.

* Your job is to complete the implementation of several filters in the `load_kernel(args)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 3.1`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

## Running, improving, and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 6 --data dreese --kernel average --display --save`<br/>
`python3 main.py --current_step 6 --data dreese --kernel binomial --display --save`<br/>
These commands will run your code. You will see several generated images, and several texts displayed in command lines. 

* You will see that while these filters remove noise, they also make the image overly blurred. Please develop a filter that can preserve the image content better. Please search **`#### Your job 3.3`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

* Once completed, please run the following command<br/>
`python3 main.py --current_step 6 --data dreese --kernel your_denoising_kernel --display --save`<br/>
These commands will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `6_Kernel_binomial.png`, `6_Kernel_your_denoising_kernel.png`, `6_Results_Kernel_binomial.npz`, `6_Results_Kernel_your_denoising_kernel.npz`, `6_Convolution_output_dreese_binomial.png`, and `6_Convolution_output_dreese_your_denoising_kernel.png`, which you will include in your submission.

* Please describe how you design your solution in the pdf.


# Question 7: (10 pts)

* Go to the `main` function and find `if int(args.current_step) == 7:` and read the corresponding code.

* Given the input image `I`, you need to perform convolution to generate the image Laplacian.

* Your job is to complete the implementation of one filter in the `load_kernel(args)` function. Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format. Please make sure that your results follow the required numpy array shapes. 

* You may search **`#### Your job 3.2`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

## Running, improving, and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 7 --data dreese --kernel Laplacian --display --save`<br/>
This command will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `7_Convolution_output_dreese_Laplacian.png`, `7_Kernel_Laplacian.png`, `7_Results_Convolution_output_dreese_Laplacian.npz`, and `7_Results_Kernel_Laplacian.npz`, which you will include in your submission.

  
# Question 8: (20 pts)

* Go to the `main` function and find `if int(args.current_step) == 8:` and read the corresponding code.
 
* In this question, there is a target image `I_target`, which is combined with other images (i.e., `I_1` and `I_2`) to become `I`. Your goal is to recover `I_target` from `I`
  
* Your job is to complete the implementation of the `Recover_function(args, I, u_target_freq, v_target_freq)` function.  Please go to the function and carefully read the input, output, and instructions. You can assume that the actual inputs will follow the input format. Please make sure that your results follow the required numpy array shapes.
  
* This question is very much open-ended, and I do expect to see different solutions and approaches from different students.

* You may search **`#### Your job 4`** to locate where to amend your implementation. You will see some instructions there. You are free to create more space in between.

## Running and saving

* Once completed, please run the following command<br/>
`python3 main.py --current_step 8 --display --save`<br/>
This command will run your code. You will see several generated images, and several texts displayed in command lines. 

* The code will generate `8_Recovered_image_none.png` and `8_Results_Recovered_image_none.npz`, which you will include in your submission.
  
* Please describe how you design your solution (and why) in the pdf. We want to learn about your design process!


# What to submit:

* Please see the beginning of the page. Please follow **Submission instructions** to submit a .zip file named name.number.zip (e.g., chao.209.zip). Failing to submit a single .zip file will not be graded.
