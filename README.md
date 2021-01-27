# Deep Learning (and Machine Learning) for Human Activity Recognition
> Keras implementation of CNN, DeepConvLSTM, and SDAE and LightGBM for sensor-based Human Activity Recognition (HAR).  

This repository contains keras (tensorflow.keras) implementation of Convolutional Neural Network (CNN) [1], Deep Convolutional LSTM (DeepConvLSTM) [1], Stacked Denoising AutoEncoder (SDAE) [2], and Light GBM for human activity recognition (HAR) using smartphones sensor dataset, *UCI smartphone* [3].

**Table 1.** The summary of the results amongst five methods on UCI smartphone dataset.  

| Method | Accuracy | Precision | Recall | F1-score |
| --- | :---: | :---: | :---: | :---: |
| LightGBM | **96.33** | **96.58** | **96.37** |  **96.43** |
| CNN [1] | 95.29 | 95.46 | 95.50 |  95.47 |
| DeepConvLSTM [1] | 95.66 | 95.71 | 95.84 | 95.72 |
| SDAE [2] | 78.28 | 78.83 | 78.47 | 78.25 |
| MLP | 93.81 | 93.97 | 94.04 |  93.85 |


# Setup
Dockerfile creates a virtual environment with Keras (tensorflow 2.3) and Python 3.8.
If you wan to use this implementation locally within a docker container:
```bash
$ git clone git@github.com:takumiw/Deep-Learning-for-Human-Activity-Recognition.git
$ cd Deep-Learning-for-Human-Activity-Recognition
$ make start-gpu
# poetry install
```

Then run code by:
```bash
# poetry run python run_cnn.py
```

# Dataset
The dataset includes types of movement amongst three static postures (*STANDING*, *SITTING*, *LYING*) and three dynamic activities (*WALKING*, *WALKING DOWNSTAIRS*, *WALKING UPSTAIRS*). 

![](https://img.youtube.com/vi/XOEN9W05_4A/0.jpg)  
[Watch video](https://www.youtube.com/watch?v=XOEN9W05_4A)  
[Download dataset](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)

# Methods
## Feature engineering + LightGBM
To represent raw sensor signals as a feature vector, 621 features are created by preprocessing, fast Fourier transform (FFT), statistics, etc. ([run_generate_features.py](https://github.com/takumiw/Deep-Learning-for-Human-Activity-Recognition/blob/master/run_generate_features.py).)  
The 621 features of the training dataset were trained on a LightGBM classifier by 5-fold cross-validation, and evaluated on the test dataset.

## Convolutional Neural network (CNN)
The preprocessed raw sensor signals are classified CNN. The CNN architecture, which is a baseline to DeepConvLSTM, follows the study [1]

## Deep Convolutional LSTM (DeepConvLSTM)
The preprocessed raw sensor signals are classified DeepConvLSTM. The fully connected layers of CNN are replaced with LSTM layers. The DeepConvLSTM architecture follows the study [1].

## Stacked Denoising AutoEncoder (SDAE)
The preprocessed raw sensor signals are trained with SDAE, softmax layer is superimposed on top of Encoder, then whole network is fine-tuned for the target classification task. I used as same settings to the reference [2] as possible, but could not reproduce the result in the paper.

## Multi Layer Perceptron (MLP)
The preprocessed raw sensor signals are trained with MLP, which consists of two hidden layers.

# Reference
[1] Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and lstm recurrent neural networks for multimodal wearable activity recognition. *Sensors*, *16*(1), 115.  
[2] Gao, X., Luo, H., Wang, Q., Zhao, F., Ye, L., & Zhang, Y. (2019). A human activity recognition algorithm based on stacking denoising autoencoder and lightGBM. *Sensors,* *19*(4), 947.  
[3] Reyes-Ortiz, J. L., Oneto, L., Samà, A., Parra, X., & Anguita, D. (2016). Transition-aware human activity recognition using smartphones. *Neurocomputing*, *171*, 754-767.

# Author
- Takumi Watanabe ([takumiw](https://github.com/takumiw))
- Sophia University