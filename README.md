# Deep Learning (Machine Learning) for Human Activity Recognition
In this repository, I implement human activity recognition (HAR) using smartphones sensor dataset. I evaluate machine learning method by feature engineering and LightGBM model, and deep learning method by using convolutional neural network (CNN) and long short-term memory (LSTM) model. I use scikit-learn libraries and Keras framework to create these models. (Please refer to [pyproject.toml](https://github.com/takumiw/Deep-Learning-for-Human-Activity-Recognition/blob/master/pyproject.toml.) 

The dataset includes the types of movement amongst three static postures, three dynamic activities, and transitions that occurred between the static postures. The six basic activities below are predicted here because the transition classes have very few samples. (Twelve activities classification is coming soon :))
- STANDING
- SITTING
- LYING
- WALKING
- WALKING DOWNSTAIRS
- WALKING UPSTAIRS

## Dataset
![](https://img.youtube.com/vi/XOEN9W05_4A/0.jpg)  
[Watch video](https://www.youtube.com/watch?v=XOEN9W05_4A)  
[Download dataset](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)

## Results
### Feature engineering + LightGBM classifier
To represent raw sensor signals as a feature vector, 621 features were created by preprocessing, fast Fourier transform (FFT), statistics, etc. Please refer to [generate_features_of_hapt.py](https://github.com/takumiw/Deep-Learning-for-Human-Activity-Recognition/blob/master/generate_features_of_hapt.py) for more details and implemented code.  
The 621 features of the training dataset were trained on a LightGBM classifier by 5-fold cross-validation, and evaluated on the test dataset. Please refer to [models/lgbm.py](https://github.com/takumiw/Deep-Learning-for-Human-Activity-Recognition/blob/master/models/lgbm.py) for more details and implemented code.  
The test scores below show very good results.
* **Accuracy: 96.37%**
* F1 score (macro avg. over six classes): 96.46
* Precision score (macro avg. over six classes): 96.57
* Recall score (macro avg. over six classes): 96.42  
The feature importance shows that time-domain features are especially effective for classifying activities.
![](https://user-images.githubusercontent.com/30923675/80077504-8c0e1100-8588-11ea-84f4-024d34fdf763.png)



### Convolutional Neural network (CNN)
Coming soon.


### Long Short-Term Memor (LSTM)
Coming soon.