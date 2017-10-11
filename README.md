# CNN-for-Biomedical-Image-Processing
Improving the quality of reconstruction image by radon transformation by CNN. The convolution neural network is coded by tensorflow. This project is a reproduction of the Deep Convolutional Neural Network for Inverse Problems in Imaging[1].

 

The architecture of the CNN is adapted from the U-net by remove the skip connection of the residual learning, and all the convolution layers are padded to retain the same size as the input tensor. Besides, the final output image is one channel, which is different from the original U-net since U-net needs to get the background and foreground.

![alt text](https://github.com/shawnye1994/CNN-for-Radon-Transformation/blob/master/CNN%20architecture.jpg)

>The training data is composed of two different data set. The first one is ellipsoid dataset. It’s a synthetic dataset that comprises 500 >images of ellipses of random intensity, size, and location. Sinograms for this data are 729 pixels by 1,000 views and are created using >the analytical expression for the X-ray transform of an ellipse. The matlab function iradon is used for FBPs.
>The second training dataset is a biomedical synthetic dataset that comprises 500 real in-vivo CT images from the Low-dose Grand challenge >competition from database made by the Mayo clinic. Sinograms for this data are 729 pixels by 1,000 views and are created using the Matlab >function radon, iradon is again used for FBPs.

>The test dataset is a real CT dataset that comprises 377 sinograms collected from an experiment at the TOMCAT beam line of the Swiss >Light Source at the Paul Scherrer Institute in Villigen, Switzerland.
>Each sinogram is 1493 pixels by 721 views and comes from one z-slice of a single rat brain.

## Reference

**[1] Jin, Kyong Hwan, et al. "Deep convolutional neural network for inverse problems in imaging." IEEE Transactions on Image Processing 26.9 (2017): 4509-4522.**

