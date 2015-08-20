# ml4j-mnist :  Classifying MNIST digits using ml4j

Kaggle Competition Entries:

* https://www.kaggle.com/c/digit-recognizer/leaderboard?submissionId=1877195  ( Feed Forward NN with weights pre-trained by Supervised Deep Belief Network )

* https://www.kaggle.com/c/digit-recognizer/leaderboard?submissionId=1880746  ( Convolutional Network, as per demo)



## Demos Provided:

* DigitImageRawDataClassifierDemo :   Classifies raw MNIST data from csv files using pre-learnt Convolutional Neural Network
* DigitImageClassifierDemo :   Classifies (28 * 28) images from jpg files using pre-learnt Convolutional Neural Network

* DigitClassificationCNNTrainer  :  Trains a Convolutional Neural Network to classify images using raw MNIST data from csv files
* DigitClassificationFNNTrainer   :  Trains a Feed Forward Neural Network to classify images using raw MNIST data from csv files

## Download/Import:

git clone https://github.com/ml4j/ml4j-mnist.git

cd ml4j-mnist

mvn eclipse:eclipse  ( To import into Eclipse )

## Running the demos in Eclipse

*  Ensure you set memory settings appropriately for training - eg.  -Xms8000M -Xmx15000M

*   JBlas is available by default on Macs - if it's not available on your system, a flag can be changed in the demo code to switch to using JAMA for matrix-matrix multiplication ( will run slower)  

*   Demos assume Cuda GPU is not available - this can be changed via flags in demo code to speed up execution of both training demos and classification demos for larger datasets  ( for smaller datasets, the overhead of bus transfer means Cuda may run slower)

*  If using Cuda GPU acceleration for the demos, ensure that DYLD_LIBRARY_PATH is available in Eclipse environment ( eg. by launching Eclipse from command line)







