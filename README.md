# Udacity_GPU_Prog
My work related to the Udacity Introduction to
Parallel Programming, CS 344. I am taking this course to improve my expertise of GPU
Parallel programming, and after nearly every set of lectures, there
are problem sets that enable the student to practice the skills
taught. Here, I have listed the assignments I've completed and brief
descriptions of the technical problems I solved. **All assignments are
applications of image processing and are implemented using OpenCV in
CUDA C/C++**.

I have uploaded this code which contains my completed solutions along
wth the rest of the code in the library that was needed to compile and
run it. However, I do not recommned trying to compile this code on its
own since it's missing some CMake directives, and this code is only
given here for demonstration purposes. To find the code that I wrote,
please look in: `Problem Set <INT>/student_func.cu`.

Udacity course that this work comes from: https://www.udacity.com/course/intro-to-parallel-programming--cs344

Code originally forked from github repo: https://github.com/udacity/cs344 (but the solutions, written up in student_func.cu are my own).

## Brief Description of Problem Sets:
* Problem Set 1: *Color to Greyscale conversion.* Converted RGBA image to greyscale using the formula recommended by the NTSC.

* Problem Set 2: *Image Blurring.* 1) Convert RGBA image that is defined as an array of structures (AoS) with a byte representing the R, G, B or A weight at each pixel to a structure of arrays (SoA) so that each channel (which is to be blurred separately) is layed out in contiguous memory. 2) Used gaussian blurring to blur/smooth each channel separately in the image. Among other things, this problem set employed the **map**/**stencil** parallel primitive algorithm.

* Problem Set 3: *HDR Tone Mapping.* The main aspect of Tone Mapping that is parallelizeable and was the focus of this problem set is the conversion of an image of luminance values at each pixel to a histogram of luminance values and then finding the cumulative distribution function (CDF) so that histogram equalization could be performed. To find the CDF, it was necessary to 1) compute the min/max of the luminance value in parallel, 2) generate a histogram of the luminance values, and 3) perform an exclusive scan on the histogram. I chose the blelloch scan to do the final step. Among other things, this assigment employed the **reduce**, **histogram**, and **scan** parallel primitive algorithms.
