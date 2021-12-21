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

**NOTE**: Two image files, [messi5.jpg](/OpenCV/messi5.jpg) and [opencv_logo.png](/OpenCV/opencv_logo.png) was used within
this program.
## Deep Learning
