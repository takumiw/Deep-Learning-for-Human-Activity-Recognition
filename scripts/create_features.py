# -*- coding:utf-8 -*-
import numpy as np
from obtain_features import ObtainFeatures


def create_features(acc_raw, gyro_raw):    
    of = ObtainFeatures(acc_raw, gyro_raw)
    
    # Remove noises by median filter & Butterworth filter
    acc_raw = of.apply_median_filter(acc_raw)
    acc_raw = of.apply_butterworth_filter(acc_raw)
    gyro_raw = of.apply_median_filter(gyro_raw)
    gyro_raw = of.apply_butterworth_filter(gyro_raw)
    
    # Sample signals in fixed-width sliding windows
    tAccXYZ, tBodyGyroXYZ = of.segment_signal(acc_raw, gyro_raw)
    
    # Separate acceleration signal into body and gravity acceleration signal
    tBodyAccXYZ, tGravityAccXYZ = [], []
    for acc in tAccXYZ:
        body_acc, grav_acc = of.remove_gravity(acc)
        tBodyAccXYZ.append(body_acc)
        tGravityAccXYZ.append(grav_acc)
    
    # Obtain Jerk signals of body linear acceleration and angular velocity
    tBodyAccJerkXYZ, tBodyGyroJerkXYZ = [], []
    for body_acc, gyro in zip(tBodyAccXYZ, tBodyGyroXYZ):
        body_acc_jerk = of.obtain_jerk_signal(body_acc)
        gyro_jerk = of.obtain_jerk_signal(gyro)
        
        tBodyAccJerkXYZ.append(body_acc_jerk)
        tBodyGyroJerkXYZ.append(gyro_jerk)
        
    # Calculate the magnitude of three-dimensional signals using the Euclidean norm
    tBodyAccMag, tGravityAccMag, tBodyAccJerkMag, tBodyGyroMag, tBodyGyroJerkMag = [], [], [], [], []
    for body_acc, grav_acc, body_acc_jerk, gyro, gyro_jerk in zip(tBodyAccXYZ, tGravityAccXYZ, tBodyAccJerkXYZ, tBodyGyroXYZ, tBodyGyroJerkXYZ):
        body_acc_mag = of.obtain_magnitude(body_acc)
        grav_acc_mag = of.obtain_magnitude(grav_acc)
        body_acc_jerk_mag = of.obtain_magnitude(body_acc_jerk)
        gyro_mag = of.obtain_magnitude(gyro)
        gyro_jerk_mag = of.obtain_magnitude(gyro_jerk)
        
        tBodyAccMag.append(body_acc_mag)
        tGravityAccMag.append(grav_acc_mag)
        tBodyAccJerkMag.append(body_acc_jerk_mag)
        tBodyGyroMag.append(gyro_mag)
        tBodyGyroJerkMag.append(gyro_jerk_mag)
        
    # Obtain amplitude spectrum using Fast Fourier Transform (FFT).
    fBodyAccXYZ, fBodyAccJerkXYZ, fBodyGyroXYZ, fBodyAccMag, fBodyAccJerkMag, fBodyGyroMag, fBodyGyroJerkMag = [], [], [] ,[] ,[] ,[], []
    for body_acc, body_acc_jerk, gyro, body_acc_mag, body_acc_jerk_mag, gyro_mag, gyro_jerk_mag in \
        zip(tBodyAccXYZ, tBodyAccJerkXYZ, tBodyGyroXYZ, tBodyAccMag, tBodyAccJerkMag, tBodyGyroMag, tBodyGyroJerkMag):
        body_acc_amp = of.obtain_amplitude_spectrum(body_acc)
        body_acc_jerk_amp = of.obtain_amplitude_spectrum(body_acc_jerk)
        gyro_amp = of.obtain_amplitude_spectrum(gyro)
        body_acc_mag_amp =  of.obtain_amplitude_spectrum(body_acc_mag)
        body_acc_jerk_mag_amp = of.obtain_amplitude_spectrum(body_acc_jerk_mag)
        gyro_mag_amp = of.obtain_amplitude_spectrum(gyro_mag)
        gyro_jerk_mag_amp = of.obtain_amplitude_spectrum(gyro_jerk_mag)
        
        fBodyAccXYZ.append(body_acc_amp)
        fBodyAccJerkXYZ.append(body_acc_jerk_amp)
        fBodyGyroXYZ.append(gyro_amp)
        fBodyAccMag.append(body_acc_mag_amp)
        fBodyAccJerkMag.append(body_acc_jerk_mag_amp)
        fBodyGyroMag.append(gyro_mag_amp)
        fBodyGyroJerkMag.append(gyro_jerk_mag_amp)
    
    #  Following signals are obtained.
    time_signals = [
        tBodyAccXYZ, 
        tGravityAccXYZ, 
        tBodyAccJerkXYZ, 
        tBodyGyroXYZ, 
        tBodyGyroJerkXYZ,
        tBodyAccMag,
        tGravityAccMag, 
        tBodyAccJerkMag, 
        tBodyGyroMag, 
        tBodyGyroJerkMag]
    freq_signals = [
        fBodyAccXYZ,
        fBodyAccJerkXYZ,
        fBodyGyroXYZ,
        fBodyAccMag,
        fBodyAccJerkMag,
        fBodyGyroMag,
        fBodyGyroJerkMag]
    all_signals = time_signals + freq_signals
    
    # Calculate feature vectors
    features = np.zeros((len(tBodyAccXYZ), 561))
    
    for i in range(len(tBodyAccXYZ)):
        feature_vector = np.array([])
        
        # mean, std, mad, max, min, sma, energy, iqr, entropy
        for t_signal in all_signals:
            sig = t_signal[i]
            mean = of.obtain_mean(sig)
            std = of.obtain_std(sig)
            mad = of.obtain_mad(sig)
            max_val = of.obtain_max(sig)
            min_val = of.obtain_min(sig)
            sma = of.obtain_sma(sig)
            energy =  of.obtain_energy(sig)
            iqr = of.obtain_iqr(sig)
            entropy = of.obtain_entropy(sig)
            
            feature_vector = np.hstack((feature_vector, mean, std, mad, max_val, min_val, sma, energy, iqr, entropy))
            
        # arCoeff
        for t_signal in time_signals:
            sig = t_signal[i]
            arCoeff = of.obtain_arCoeff(sig)
            
            feature_vector = np.hstack((feature_vector, arCoeff))
        
        # correlation
        for t_signal in [tBodyAccXYZ, tGravityAccXYZ, tBodyAccJerkXYZ, tBodyGyroXYZ, tBodyGyroJerkXYZ]:
            sig = t_signal[i]
            correlation = of.obtain_correlation(sig)
            
            feature_vector = np.hstack((feature_vector, correlation))
        
        # maxInds, meanFreq, skewness, kurtosis
        for t_signal in freq_signals:
            sig = t_signal[i]
            maxInds = of.obtain_maxInds(sig)
            meanFreq = of.obtain_meanFreq(sig)
            skewness = of.obtain_skewness(sig)
            kurtosis = of.obtain_kurtosis(sig)
            
            feature_vector = np.hstack((feature_vector, maxInds, meanFreq, skewness, kurtosis))
        
        # bandsEnergy
        for t_signal in [tBodyAccXYZ, tBodyAccJerkXYZ, tBodyGyroXYZ]:
            sig = t_signal[i]
            bandsEnergy = of.obtain_bandsEnergy(sig)
            
            feature_vector = np.hstack((feature_vector, bandsEnergy))
        
        # angle
        gravityMean = tGravityAccXYZ[i].mean()
        tBodyAccMean = tBodyAccXYZ[i].mean()
        tBodyAccJerkMean = tBodyAccJerkXYZ[i].mean()
        tBodyGyroMean = tBodyGyroXYZ[i].mean()
        tBodyGyroJerkMean = tBodyGyroJerkXYZ[i].mean()
        tXAxisAcc = tAccXYZ[i]['x']
        tXAxisGravity = tGravityAccXYZ[i]['x']
        tYAxisAcc = tAccXYZ[i]['y']
        tYAxisGravity = tGravityAccXYZ[i]['y']
        tZAxisAcc = tAccXYZ[i]['z']
        tZAxisGravity = tGravityAccXYZ[i]['z']
        
        tBodyAccWRTGravity = of.obtain_angle(tBodyAccMean, gravityMean)
        tBodyAccJerkWRTGravity = of.obtain_angle(tBodyAccJerkMean, gravityMean)
        tBodyGyroWRTGravity = of.obtain_angle(tBodyGyroMean, gravityMean)
        tBodyGyroJerkWRTGravity = of.obtain_angle(tBodyGyroJerkMean, gravityMean)
        tXAxisAccWRTGravity = of.obtain_angle(tXAxisAcc, tXAxisGravity)
        tYAxisAccWRTGravity = of.obtain_angle(tYAxisAcc, tYAxisGravity)
        tZAxisAccWRTGravity = of.obtain_angle(tZAxisAcc, tZAxisGravity)
        
        feature_vector = np.hstack((feature_vector, tBodyAccWRTGravity, tBodyAccJerkWRTGravity, tBodyGyroWRTGravity, tBodyGyroJerkWRTGravity, \
                                   tXAxisAccWRTGravity, tYAxisAccWRTGravity, tZAxisAccWRTGravity))
    
        features[i] = feature_vector
    
    return features