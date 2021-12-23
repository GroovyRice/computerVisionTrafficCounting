# Traffic Counting with Computer Vision
Currently, this repository consists of 2 separated branches
- [Main_Files](#Main-Files-for-Traffic-Counting): This section houses all the files that make the program run. Different versions will be categorised into folders.
- [Learning](#Learning-OpenCV-and-Deep-Learning-with-Python): This section contains various files from exemplar code relating to OpenCV and Deep Learning practise. It allows for easy reflection.
The footage of _Traffic_ can be found [here](https://drive.google.com/drive/folders/1VTXwcydJPd81ZAMDuM_sng3yKgEDluhB?usp=sharing)

## REQUIRED
**Python**: install [here](https://www.python.org/downloads/)

**PyCharm**: this IDE will allow you to easily pull the Git repository, alternatively you can just use it to compile within a separate _Project_. This can be installed from [here](https://www.jetbrains.com/pycharm/)

**Numpy**: after installing Python run this in _Command Prompt_ `pip install numpy` makesure you have version 1.21.4 check this in the Python console by typing

> `import numpy as np`<br/>
> `print(np.__version__)`

**OpenCV**: similarly `pip install opencv-contrib-python` if this doesn't work just use `pip install opencv-python`<br/>
If you struggle installing please don't hesitate to email me.


# Learning OpenCV and Deep Learning with Python
For learning purposes I have structured this document to contain everything I have learned and where it can be
seen within various files. This will be split into a multitude of different parts quick access to these parts can be
found here:
1) [OpenCV](#OpenCV)
2) [Deep Learning](#Deep-Learning)
## OpenCV
[**Grayscale.py**](/OpenCV/grayscale.py): Creates a 200 x 200, BGR (OpenCV stores RGB as BGR) with the top half and
bottom half being two separate colours. These colours are then converted to grayscale using OpenCVs `cvtColor()` function.
After converting the threshold function from the same library is used then the output displayed.
The purpose of this program is to test the various different types of thresholds and get a better grasp
on how they work on the image. The most common is if the grayscale value is within the bounds of the threshold
then it that value would be changed to the maximum value otherwise it will go to zero or stay the same. These different
types are explained in better detail within the file.<br/>
<br/>
[**Image-operations.py**](/OpenCV/imageoperations.py): This one loads two separate images and then joins them together using
various functions within OpenCVs arsenal. The purpose of this is to grasp a better understanding of how masks work and how
the arrays can be manipulated using various tools. Additionally, using the `getTickCount()` function the time taken to undergo
these operations was recorded and compared to other tools like _NumPy_ to see which process is ultimately faster. The result 
was that _OpenCV_ is hands down always faster than _NumPy_ so if *required* always use OpenCV for image operations. 

**NOTE**: Two image files, [messi5.jpg](/OpenCV/messi5.jpg) and [opencv_logo.png](/OpenCV/opencv_logo.png) were used within
this program.
## Deep Learning
### IMDB Movie Review Classifier
[**imdb_1.py**](/Deep_Learning/imdb_1.py): This is the initial part of a program that will classify IMDB movie reviews as
either positive or negative. As this is for learning, KERAS has the datasets of reviews and whether they were positive
or negative already sorted as two separate lists; the first contains data for the most common words used within each
review. This means that there is a specific list with what common words were used for each individual review, the second
contains a list for whether it was positive or negative review (0 or 1). Within this dataset there was 25 thousand
different reviews and 10 thousand additional test data and labels. _IMDB (1)_ shows how to import this dataset and also
explains a very useful technique used in **for loops** and **matrices**.<br/>
[**imdb_2.py**](/Deep_Learning/imdb_2.py): Continuing from _IMDB (1)_ this also teaches a very useful way to use **for loops**
with an enumerator, much like the likes found in C-programming. It also continues with useful **matrix** manipulation
followed by the general process of setting up a model, compiling and validating within KERAS. It uses the 
Matplotlib module to verify how validate the training of the model went. This specific dataset and training ended
with it _overfitting_ the data.<br/>
[**imdb_3.py**](/Deep_Learning/imdb_3.py): This wraps up the trilogy of files for the IMDB classification task, using the
knowledge acquired from _IMDB (2)_ reducing the EPOCHS to 4 to prevent the model from _overfitting_ and then passing
in new data to predict whether the review was positive or negative. The model is very confident for specific reviews
which result in a 0.99 or more, or 0.1; however, it is less confident other times (0.6, 0.4).<br/>
<br/>
Ultimately, this was a great fundamental lesson into _Deep Learning_ however there were a multitude of
various hurdles, software and hardware based namely installing CUDA and cudNN integration with NVIDIA GPUs. As it stands
on the _23/12/21_ this might not be a viable approach for **Traffic Counting** due to the hardware requirements. Further
research into _Deep Learning_ and its uses in **Cloud Computing** might be the approach.
# Main Files for Traffic Counting
