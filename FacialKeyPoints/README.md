# Deep learning for facial keypoints detection
# A kaggle competition ranked at #22. 

![alert] (intro.png)

How the CNN built and trained was followed by [Daniel Nouri's neural network tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/) and the cascade part(Net 5) was inspired by the paper ["Deep Convolutional Network Cascade for Facial Point Detection"](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sun_Deep_Convolutional_Network_2013_CVPR_paper.pdf).

Brief suggestion: a) if want to train the nets on your own, run ‘Net1’ - ‘Net5’; b) if want to check the recognition result, run ‘test2’; c) if want to check the history training or valid loss curve, run ‘test3’. 


Please make sure reading the this file before running the codes.

1, These libraries must be installed: theano, numpy, matplotlib, pandas, lasagne 

2, If the device running these code has GPU, then either CUDA or OpenCL should be confirmed able to work smoothly with theano to make sure the code won’t take forever to run (with gpu each epoch takes 1-9s according to different nets, without gnu each epoch takes 2min-5min according to different nets, basically training each net at least needs 1000-3000 epochs)

3, ‘load’ file contains the functions defined how to loading the images from the provided csv to our working space and how to load the specific areas which are needed in cascade CNN either for training or testing. 

4, ‘Net1’ file is the simple one layer neural network to distinguish which preprocessing method is better

5, ‘Net2’ file is the one layer neural network with input augmentation

6, ’Net3’ file is the basic CNN

7, ‘Net4’ file is the improved CNN with dropout and changeable learning rate as well as momentum.

8, ‘Net5’ file is the last step in training cascade CNN specialists nets. It shows how the special areas (left eye, right eye, nose, mouse) are trained. 

9, ‘test1’ file draw random 8 images and use cascade CNN doing prediction for them. (only using level 1 and level 2, running the whole cascade CNN please train the all specialists in Net5 and when finished it will show the result and automatically create a submission file for Kaggle competition) 

10, ‘test2’ file is to draw error plot for the already trained nets.
