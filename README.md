# image-text-recognition

Region-based Convolutional Neural Networks for Text Detection in Images

Written by : Taehoon Kim, jamesk1228@gmail.com, Sogang University, Seoul, Republic of Korea

Objective: Detect digits from black & white images.

Prerequsites: python3, tensorflow, scikit-image, numpy, opencv-python (Recommand using anaconda)

How-to: 
1. Clone this git repository 
2. For pre-trained MNIST Handwritten digit Convolutional Neural Network model, 
   go to https://drive.google.com/open?id=0B7TzarHysU5fNXlwMF9Cb0RTV1U
3. Unzip pre-trained model and place it under root git repository.
4. Place your custom image in "images/"
5. To run the code in Jupyter Notebook, use main.ipynb instead of main.py.
6. You can find our raw value and coordinates of labels in patent image from "labels/image_name_label.txt"

Corresponding Paper: on process

References:

https://github.com/FraPochetti/ImageTextRecognition

https://github.com/oliviersoares/mnist

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373 (link is external)

Christoph Riedl, Richard Zanibbi, Marti A. Hearst, Siyu Zhu, Michael Menietti, Jason Crusan, Ivan Metelsky, Karim R. Lakhani, Detecting figures and part labels in patents: competition-based development of graphics recognition algorithms International Journal on Document Analysis and Recognition (IJDAR), Volume 19, Issue 2, pp. 155–172, 2016

Ross Girshick, Jeff Donahue, Trevor Darrell, Region-Based Convolutional Networks for Accurate Object Detection and Segmentation, IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume: 38, Issue 1, pp. 142–158, 2015

Max Jaderberg, Karen Simonyan, Andrea Vedaldi, Andrew Zisserman, Reading Text in the Wild with Convolutional Neural Networks, International Journal of Computer Vision, Volume 116, Issue 1, pp. 1-20, 2016

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, Going Deeper with Convolutions, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1-9, 2015

Xiang Zhang, Junbo Zhao, Yann LeCun, Character-level Convolutional Networks for Text Classification, Advances in Neural Information Processing Systems 28 (NIPS 2015)

Nobuyuki Otsu, "A threshold selection method from gray-level histograms". IEEE Trans. Sys., Man., Cyber. 9 (1), pp. 62–66, 1979


