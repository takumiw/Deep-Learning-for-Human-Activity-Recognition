# -*- coding:utf-8 -*-
import numpy as np
from numpy.linalg import norm
import pandas as pd
import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from scipy import signal, stats
from statsmodels.regression.linear_model import burg
from statsmodels.distributions.empirical_distribution import ECDF


class ObtainFeatures():
    fs = 50  # Sampling frequency of sensor data.
    win_size = 128  # WIndow size of sliding window to segment raw signals.
    overlap_rate = 0.5  # Overlap rate of sliding window to segment raw signals.
    win_sec = 2.56  # Elapsing time (second) of window. 
    
    
    def __init__(self, acc, gyro):
        """
        Args:
            acc (DataFrame): Accelerometer 3-axial raw signals
            gyro (DataFrame): Gyroscope 3-axial raw signals
        """
        self.acc = acc
        self.gyro = gyro
        self.acc_seg = None
        self.gyro_seg = None
        
        
    def apply_median_filter(self, sensor_signal):
        """
        A median filter with window length 5 is applied to remove noise in signals.
        Args:
            sensor_signal (DataFrame): Raw signal
        Returns:
            retval (DataFrame): Median filtered signal
        """
        return sensor_signal.rolling(window=5, center=True, min_periods=1).median()
        
        
    def apply_butterworth_filter(self, sensor_signal):
        """
        A 3rd order low pass Butterworth filter with a corner frequency of 20 Hz is applied to remove noise in signals.
        Args:
            sensor_signal (DataFrame): Raw signal
        Returns:
            retval (DataFrame): Butterworth filtered signal
        """       
        fc = 20  # cutoff frequency
        w = fc / (self.__class__.fs / 2)  # Normalize the frequency
        b, a = signal.butter(3, w, 'low')  # 3rd order low pass Butterworth filter
        
        return pd.DataFrame(signal.filtfilt(b, a, sensor_signal, axis=0), columns=['x', 'y', 'z'])  # Apply Butterworth filter
       
    
    def segment_signal(self, acc, gyro):
        """
        Sample sensor signals in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window).
        Args:
            acc (DataFrame): Raw acceleration signal
            gyro (DataFrame): Raw gyro signal
        Returns:
            acc_seg (list): List of segmented acceleration sigmal. Element is a segmented DataFrame.
            gyro_seg (list): List of segmented gyro sigmal. Element is a segmented DataFrame.
        """
        assert len(acc) == len(gyro), f'Length of acceleration signal ({len(acc)}) does not match to length of gyro signal ({len(gyro)}).'
        
        win_size = self.__class__.win_size
        overlap_rate = self.__class__.overlap_rate
        acc_seg, gyro_seg = [], []
        
        for start_idx in range(0, len(acc) - win_size, int(win_size * overlap_rate)):
            acc_seg.append(acc.iloc[start_idx:start_idx + win_size].reset_index(drop=True))
            gyro_seg.append(gyro.iloc[start_idx:start_idx + win_size].reset_index(drop=True))

        return acc_seg, gyro_seg
        
    
    def remove_gravity(self, acc):
        """
        Separate acceleration signal into body and gravity acceleration signal.
        Another low pass Butterworth filter with a corner frequency of 0.3 Hz is applied.
        Args:
            acc (DataFrame): Segmented acceleration signal
        Returns:
            acc_body (DataFrame): Body acceleration signal
            acc_grav (DataFrame): Gravity acceleration signal
        """
        fc = 0.3  # cutoff frequency
        w = fc / (self.__class__.fs / 2)  # Normalize the frequency
        b, a = signal.butter(3, w, 'low')  # 3rd order low pass Butterworth filter
        
        acc_grav = pd.DataFrame(signal.filtfilt(b, a, acc, axis=0), columns=['x', 'y', 'z'])  # Apply Butterworth filter
        
        # Substract gravity acceleration from acceleration sigal.
        acc_body = acc - acc_grav
        return acc_body, acc_grav
    
    
    def obtain_jerk_signal(self, sensor_signal):
        """
        Derive signal to obtain Jerk signals
        Args:
            sensor_signal (DataFrame)
        Returns:
            jerk_signal (DataFrame):
        """
        jerk_signal = sensor_signal.diff(periods=1)  # Calculate difference 
        jerk_signal.iloc[0] = jerk_signal.iloc[1]  # Fillna
        jerk_signal = jerk_signal / (1 / self.__class__.fs)  # Derive in time (1 / sampling frequency)
        return jerk_signal
    
    
    def obtain_magnitude(self, sensor_signal):
        """
        Calculate the magnitude of these three-dimensional signals using the Euclidean norm
        Args:
            sensor_signal (DataFrame): Three-dimensional signals
        Returns:
            retval (array): Magnitude of three-dimensional signals
        """
        return  norm(sensor_signal, ord=2, axis=1)
    
    
    def obtain_amplitude_spectrum(self, sensor_signal):
        """
        Obtain amplitude spectrum using Fast Fourier Transform (FFT).
        Args:
            sensor_signal (DataFrame): Time domain signals
        Returns:
            amp (array): Amplitude spectrum
        """
        N = len(sensor_signal)
        # dim = sensor_signal.shape[1]
        win = np.hamming(N)  # hamming window
        
        if len(sensor_signal.shape) == 1:
            sensor_signal = sensor_signal * win  # Apply hamming window
            F = np.fft.fft(sensor_signal, axis=0)  # Apply FFT
            F = F[:N//2]  # Remove the overlapping part

            amp = np.abs(F)  # Obtain the amplitude spectrum
            amp = amp / N * 2
            amp[0] = amp[0] / 2
        
        else:
            df = pd.DataFrame(columns=['x', 'y', 'z'])
            for axis in ['x', 'y', 'z']:
                df[axis] = sensor_signal[axis] * win  # Apply hamming window

            F = np.fft.fft(df, axis=0)  # Apply FFT
            F = pd.DataFrame(F, columns=['x', 'y', 'z'])  # Convert array to DataFrame
            F = F[:N//2]  # Remove the overlapping part

            amp = np.abs(F)  # Obtain the amplitude spectrum
            amp = pd.DataFrame(amp, columns=['x', 'y', 'z'])  # Convert array to DataFrame

            amp = amp / N * 2
            amp.iloc[0] = amp.iloc[0] / 2
        
        return amp
    

    def obtain_ecdf_percentile(self, sensor_signal, n_bins=10):
        """Obtain ECDF (empirical cumulative distribution function) percentile values.
        Args:
            sensor_signal (DataFrame): Time domain signals
            n_bins (int, default: 10): How many percentiles to use as a feature
        Returns:
            features (array): ECDF percentile values.
        """
        idx = np.linspace(0, sensor_signal.shape[0]-1, n_bins)  # Take n_bins linspace percentile.
        idx = [int(Decimal(str(ix)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) for ix in idx]
        features = np.array([])
        for axis in ['x', 'y', 'z']:
            ecdf = ECDF(sensor_signal[axis].values)  # fit
            x = ecdf.x[1:]  # Remove -inf
            feat = x[idx]
            features = np.hstack([features, feat])

        return features
    
    
    def obtain_mean(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:    return sensor_signal.mean()
        else:    return sensor_signal.mean().values
    
    def obtain_std(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:    return sensor_signal.std()
        else:    return sensor_signal.std().values
    
    def obtain_mad(self, sensor_signal) -> np.ndarray:
        return stats.median_absolute_deviation(sensor_signal, axis=0)
    
    def obtain_max(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:    return sensor_signal.max()
        else:    return sensor_signal.max().values
    
    def obtain_min(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:    return sensor_signal.min()
        else:    return sensor_signal.min().values
    
    def obtain_sma(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:    return (sensor_signal.sum() - self.obtain_min(sensor_signal) * len(sensor_signal)) / self.__class__.win_sec
        else:    return sum(sensor_signal.sum().values - self.obtain_min(sensor_signal) * len(sensor_signal)) / self.__class__.win_sec
    
    def obtain_energy(self, sensor_signal) -> np.ndarray:
        return  norm(sensor_signal, ord=2, axis=0) ** 2 / len(sensor_signal)
    
    def obtain_iqr(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:
            q25, q75 = np.percentile(sensor_signal, q=[25, 75])
            return q75 - q25
        else:    return  sensor_signal.quantile(.75).values - sensor_signal.quantile(.25).values
    
    def obtain_entropy(self, sensor_signal) -> np.ndarray:
        sensor_signal = sensor_signal - sensor_signal.min()
        return  stats.entropy(sensor_signal)

    def obtain_arCoeff(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:  # Signal dimension is 1
            arCoeff, _ = burg(sensor_signal, order=4)
        else:  # Signal dimension is 3
            x, _ = burg(sensor_signal['x'], order=4)
            y, _ = burg(sensor_signal['y'], order=4)
            z, _ = burg(sensor_signal['z'], order=4)
            arCoeff = np.hstack((x, y, z))
        return arCoeff
    
    def obtain_correlation(self, sensor_signal) -> np.ndarray:
         if len(sensor_signal.shape) == 1:  # Signal dimension is 1
            correlation = np.array([])
         else:  # Signal dimension is 3
            xy = np.corrcoef(sensor_signal['x'], sensor_signal['y'])[0][1]
            yz = np.corrcoef(sensor_signal['y'], sensor_signal['z'])[0][1]
            zx = np.corrcoef(sensor_signal['z'], sensor_signal['x'])[0][1]
            correlation = np.hstack((xy, yz, zx))
         return correlation

    def obtain_maxInds(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:    return np.argmax(sensor_signal)
        return sensor_signal.idxmax().values
        
    def obtain_meanFreq(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:  # Signal dimension is 1
            meanFreq = np.mean(sensor_signal * np.arange(len(sensor_signal)))
        else:  # Signal dimension is 3
            x = np.mean(sensor_signal['x'] * sensor_signal.index.values)
            y = np.mean(sensor_signal['y'] * sensor_signal.index.values)
            z = np.mean(sensor_signal['z'] * sensor_signal.index.values)
            meanFreq = np.hstack((x, y, z))
        return meanFreq
    
    def obtain_skewness(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:
            sensor_signal = pd.DataFrame(sensor_signal)
        return sensor_signal.skew().values
    
    def obtain_kurtosis(self, sensor_signal) -> np.ndarray:
        if len(sensor_signal.shape) == 1:
            sensor_signal = pd.DataFrame(sensor_signal)
        return sensor_signal.kurt().values
    
    def obtain_bandsEnergy(self, sensor_signal) -> np.ndarray:
        bandsEnergy = np.array([])
        bins =  [0, 4, 8, 12, 16, 20, 24, 29, 34, 39, 44, 49, 54, 59, 64]
        for i in range(len(bins) - 1):
            df = sensor_signal.iloc[bins[i]:bins[i+1]]
            arr = self.obtain_energy(df)
            bandsEnergy = np.hstack((bandsEnergy, arr))
        return bandsEnergy

    def obtain_angle(self, v1, v2) -> np.ndarray:
        length = lambda v: math.sqrt(np.dot(v, v))
        return math.acos(np.dot(v1, v2) / (length(v1) * length(v2)))