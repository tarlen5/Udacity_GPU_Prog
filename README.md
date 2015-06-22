# Udacity_GPU_Prog
My work related to the Udacity Introduction to Parallel Programming, CS 344. I am taking this course to improve my expertise of GPU Parallel programming, and after nearly every set of lectures, there are problem sets that enable the student to practice the skills taught. Here, I have listed the assignments I've completed and brief descriptions of the technical problems I solved. **All assignments are applications of image processing and are implemented using OpenCV in CUDA C/C++**.

Udacity course that this work comes from: https://www.udacity.com/course/intro-to-parallel-programming--cs344
Code originally forked from github repo: https://github.com/udacity/cs344

## Brief Description of Problem Sets:
* Problem Set 1: *Color to Greyscale conversion.* Converted RGBA image to greyscale using the formula recommended by the NTSC.

* Problem Set 2: *Image Blurring.* 1) Convert RGBA image that is defined as an array of structures (AoS) with a byte representing the R, G, B or A intensity at each pixel to a structure of arrays (SoA) so that each channel (which is to be blurred separately) is layed out in contiguous memory. 2) Used gaussian blurring to blur/smooth each channel separately in the image.

* Problem Set 3: *HDR Tone Mapping.* 
