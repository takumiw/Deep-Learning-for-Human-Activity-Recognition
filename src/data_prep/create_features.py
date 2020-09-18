from typing import List

import numpy as np
import pandas as pd

from src.data_prep.preprocessing import Preprocess  # Load class for obtaining features


def create_features(acc_raw: pd.DataFrame, gyro_raw: pd.DataFrame) -> np.ndarray:
    """Create features from raw acceleration and gyroscope sensor data
    Args:
        acc_raw (pd.DataFrame): Raw 3-axial accelerometer signals with columns denoting axes.
        gyro_raw (pd.DataFrame): Raw 3-axial gyroscope signals with columns denoting axes.
    Returns:
        features (np.ndarray): Created features corresponding args with columns denoting feature names.
    """
    of = Preprocess(fs=50)  # Create an instance.

    # Remove noises by median filter & Butterworth filter
    acc_raw = of.apply_filter(signal=acc_raw, filter="median", window=5)
    acc_raw = of.apply_filter(signal=acc_raw, filter="butterworth")
    gyro_raw = of.apply_filter(signal=gyro_raw, filter="median", window=5)
    gyro_raw = of.apply_filter(signal=gyro_raw, filter="butterworth")

    # Sample signals in fixed-width sliding windows
    tAccXYZ = of.segment_signal(acc_raw, window_size=128, overlap_rate=0.5, res_type="dataframe")
    tBodyGyroXYZ = of.segment_signal(
        gyro_raw, window_size=128, overlap_rate=0.5, res_type="dataframe"
    )

    # Separate acceleration signal into body and gravity acceleration signal
    tBodyAccXYZ, tGravityAccXYZ = [], []
    for acc in tAccXYZ:
        body_acc, grav_acc = of.separate_gravity(acc.copy())
        tBodyAccXYZ.append(body_acc)
        tGravityAccXYZ.append(grav_acc)

    # Obtain Jerk signals of body linear acceleration and angular velocity
    tBodyAccJerkXYZ, tBodyGyroJerkXYZ = [], []
    for body_acc, gyro in zip(tBodyAccXYZ, tBodyGyroXYZ):
        body_acc_jerk = of.obtain_jerk_signal(body_acc.copy())
        gyro_jerk = of.obtain_jerk_signal(gyro.copy())

        tBodyAccJerkXYZ.append(body_acc_jerk)
        tBodyGyroJerkXYZ.append(gyro_jerk)

    # Calculate the magnitude of three-dimensional signals using the Euclidean norm
    tBodyAccMag, tGravityAccMag, tBodyAccJerkMag, tBodyGyroMag, tBodyGyroJerkMag = (
        [],
        [],
        [],
        [],
        [],
    )
    for body_acc, grav_acc, body_acc_jerk, gyro, gyro_jerk in zip(
        tBodyAccXYZ, tGravityAccXYZ, tBodyAccJerkXYZ, tBodyGyroXYZ, tBodyGyroJerkXYZ
    ):
        body_acc_mag = of.obtain_magnitude(body_acc.copy())
        grav_acc_mag = of.obtain_magnitude(grav_acc.copy())
        body_acc_jerk_mag = of.obtain_magnitude(body_acc_jerk.copy())
        gyro_mag = of.obtain_magnitude(gyro.copy())
        gyro_jerk_mag = of.obtain_magnitude(gyro_jerk.copy())

        tBodyAccMag.append(body_acc_mag)
        tGravityAccMag.append(grav_acc_mag)
        tBodyAccJerkMag.append(body_acc_jerk_mag)
        tBodyGyroMag.append(gyro_mag)
        tBodyGyroJerkMag.append(gyro_jerk_mag)

    # Obtain amplitude spectrum using Fast Fourier Transform (FFT).
    (
        fBodyAccXYZAmp,
        fBodyAccJerkXYZAmp,
        fBodyGyroXYZAmp,
        fBodyAccMagAmp,
        fBodyAccJerkMagAmp,
        fBodyGyroMagAmp,
        fBodyGyroJerkMagAmp,
    ) = ([], [], [], [], [], [], [])
    (
        fBodyAccXYZPhs,
        fBodyAccJerkXYZPhs,
        fBodyGyroXYZPhs,
        fBodyAccMagPhs,
        fBodyAccJerkMagPhs,
        fBodyGyroMagPhs,
        fBodyGyroJerkMagPhs,
    ) = ([], [], [], [], [], [], [])
    for (
        body_acc,
        body_acc_jerk,
        gyro,
        body_acc_mag,
        body_acc_jerk_mag,
        gyro_mag,
        gyro_jerk_mag,
    ) in zip(
        tBodyAccXYZ,
        tBodyAccJerkXYZ,
        tBodyGyroXYZ,
        tBodyAccMag,
        tBodyAccJerkMag,
        tBodyGyroMag,
        tBodyGyroJerkMag,
    ):
        body_acc_amp, body_acc_phase = of.obtain_spectrum(body_acc.copy())
        body_acc_jerk_amp, body_acc_jerk_phase = of.obtain_spectrum(body_acc_jerk.copy())
        gyro_amp, gyro_phase = of.obtain_spectrum(gyro.copy())
        body_acc_mag_amp, body_acc_mag_phase = of.obtain_spectrum(body_acc_mag.copy())
        body_acc_jerk_mag_amp, body_acc_jerk_mag_phase = of.obtain_spectrum(
            body_acc_jerk_mag.copy()
        )
        gyro_mag_amp, gyro_mag_phase = of.obtain_spectrum(gyro_mag.copy())
        gyro_jerk_mag_amp, gyro_jerk_mag_phase = of.obtain_spectrum(gyro_jerk_mag.copy())

        fBodyAccXYZAmp.append(body_acc_amp)
        fBodyAccJerkXYZAmp.append(body_acc_jerk_amp)
        fBodyGyroXYZAmp.append(gyro_amp)
        fBodyAccMagAmp.append(body_acc_mag_amp)
        fBodyAccJerkMagAmp.append(body_acc_jerk_mag_amp)
        fBodyGyroMagAmp.append(gyro_mag_amp)
        fBodyGyroJerkMagAmp.append(gyro_jerk_mag_amp)

        fBodyAccXYZPhs.append(body_acc_phase)
        fBodyAccJerkXYZPhs.append(body_acc_jerk_phase)
        fBodyGyroXYZPhs.append(gyro_phase)
        fBodyAccMagPhs.append(body_acc_mag_phase)
        fBodyAccJerkMagPhs.append(body_acc_jerk_mag_phase)
        fBodyGyroMagPhs.append(gyro_mag_phase)
        fBodyGyroJerkMagPhs.append(gyro_jerk_mag_phase)

    #  Following signals are obtained by implementing above functions.
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
        tBodyGyroJerkMag,
    ]
    freq_signals = [
        fBodyAccXYZAmp,
        fBodyAccJerkXYZAmp,
        fBodyGyroXYZAmp,
        fBodyAccMagAmp,
        fBodyAccJerkMagAmp,
        fBodyGyroMagAmp,
        fBodyGyroJerkMagAmp,
        fBodyAccXYZPhs,
        fBodyAccJerkXYZPhs,
        fBodyGyroXYZPhs,
        fBodyAccMagPhs,
        fBodyAccJerkMagPhs,
        fBodyGyroMagPhs,
        fBodyGyroJerkMagPhs,
    ]

    all_signals = time_signals + freq_signals

    # Calculate feature vectors by using signals
    features = []

    for i in range(len(tBodyAccXYZ)):
        feature_vector = np.array([])

        # mean, std, mad, max, min, sma, energy, iqr, entropy
        for t_signal in all_signals:
            sig = t_signal[i].copy()
            mean = of.obtain_mean(sig)
            std = of.obtain_std(sig)
            mad = of.obtain_mad(sig)
            max_val = of.obtain_max(sig)
            min_val = of.obtain_min(sig)
            sma = of.obtain_sma(sig)
            energy = of.obtain_energy(sig)
            iqr = of.obtain_iqr(sig)
            entropy = of.obtain_entropy(sig)
            feature_vector = np.hstack(
                (feature_vector, mean, std, mad, max_val, min_val, sma, energy, iqr, entropy)
            )

        # arCoeff
        for t_signal in time_signals:
            sig = t_signal[i].copy()
            arCoeff = of.obtain_arCoeff(sig)
            feature_vector = np.hstack((feature_vector, arCoeff))

        # correlation
        for t_signal in [
            tBodyAccXYZ,
            tGravityAccXYZ,
            tBodyAccJerkXYZ,
            tBodyGyroXYZ,
            tBodyGyroJerkXYZ,
        ]:
            sig = t_signal[i].copy()
            correlation = of.obtain_correlation(sig)
            feature_vector = np.hstack((feature_vector, correlation))

        # maxInds, meanFreq, skewness, kurtosis
        for t_signal in freq_signals:
            sig = t_signal[i].copy()
            maxInds = of.obtain_maxInds(sig)
            meanFreq = of.obtain_meanFreq(sig)
            skewness = of.obtain_skewness(sig)
            kurtosis = of.obtain_kurtosis(sig)
            feature_vector = np.hstack((feature_vector, maxInds, meanFreq, skewness, kurtosis))

        # bandsEnergy
        for t_signal in [tBodyAccXYZ, tBodyAccJerkXYZ, tBodyGyroXYZ]:
            sig = t_signal[i].copy()
            bandsEnergy = of.obtain_bandsEnergy(sig)
            feature_vector = np.hstack((feature_vector, bandsEnergy))

        # angle
        gravityMean = tGravityAccXYZ[i].mean()
        tBodyAccMean = tBodyAccXYZ[i].mean()
        tBodyAccJerkMean = tBodyAccJerkXYZ[i].mean()
        tBodyGyroMean = tBodyGyroXYZ[i].mean()
        tBodyGyroJerkMean = tBodyGyroJerkXYZ[i].mean()
        tXAxisAcc = tAccXYZ[i]["x"]
        tXAxisGravity = tGravityAccXYZ[i]["x"]
        tYAxisAcc = tAccXYZ[i]["y"]
        tYAxisGravity = tGravityAccXYZ[i]["y"]
        tZAxisAcc = tAccXYZ[i]["z"]
        tZAxisGravity = tGravityAccXYZ[i]["z"]

        tBodyAccWRTGravity = of.obtain_angle(tBodyAccMean, gravityMean)
        tBodyAccJerkWRTGravity = of.obtain_angle(tBodyAccJerkMean, gravityMean)
        tBodyGyroWRTGravity = of.obtain_angle(tBodyGyroMean, gravityMean)
        tBodyGyroJerkWRTGravity = of.obtain_angle(tBodyGyroJerkMean, gravityMean)
        tXAxisAccWRTGravity = of.obtain_angle(tXAxisAcc, tXAxisGravity)
        tYAxisAccWRTGravity = of.obtain_angle(tYAxisAcc, tYAxisGravity)
        tZAxisAccWRTGravity = of.obtain_angle(tZAxisAcc, tZAxisGravity)

        feature_vector = np.hstack(
            (
                feature_vector,
                tBodyAccWRTGravity,
                tBodyAccJerkWRTGravity,
                tBodyGyroWRTGravity,
                tBodyGyroJerkWRTGravity,
                tXAxisAccWRTGravity,
                tYAxisAccWRTGravity,
                tZAxisAccWRTGravity,
            )
        )

        # ECDF
        for t_signal in [tBodyAccXYZ, tBodyGyroXYZ]:
            sig = t_signal[i].copy()
            ecdf = of.obtain_ecdf_percentile(sig)
            feature_vector = np.hstack((feature_vector, ecdf))

        features.append(feature_vector)

    return np.array(features)


def get_feature_names() -> List[str]:
    """Get feature names
    Returns:
        feature_names (List[str]): Title of features
    """
    time_signal_names = [
        "tBodyAccXYZ",
        "tGravityAccXYZ",
        "tBodyAccJerkXYZ",
        "tBodyGyroXYZ",
        "tBodyGyroJerkXYZ",
        "tBodyAccMag",
        "tGravityAccMag",
        "tBodyAccJerkMag",
        "tBodyGyroMag",
        "tBodyGyroJerkMag",
    ]
    freq_signal_names = [
        "fBodyAccXYZAmp",
        "fBodyAccJerkXYZAmp",
        "fBodyGyroXYZAmp",
        "fBodyAccMagAmp",
        "fBodyAccJerkMagAmp",
        "fBodyGyroMagAmp",
        "fBodyGyroJerkMagAmp",
        "fBodyAccXYZPhs",
        "fBodyAccJerkXYZPhs",
        "fBodyGyroXYZPhs",
        "fBodyAccMagPhs",
        "fBodyAccJerkMagPhs",
        "fBodyGyroMagPhs",
        "fBodyGyroJerkMagPhs",
    ]
    all_signal_names = time_signal_names + freq_signal_names
    feature_names = []

    for name in all_signal_names:
        for s in ["Mean", "Std", "Mad", "Max", "Min", "Sma", "Energy", "Iqr", "Entropy"]:
            if s == "Sma":
                feature_names.append(f"{name}{s}")
                continue
            if "XYZ" in name:
                n = name.replace("XYZ", "")
                feature_names += [f"{n}{s}-{ax}" for ax in ["X", "Y", "Z"]]
            else:
                feature_names.append(f"{name}{s}")

    for name in time_signal_names:
        if "XYZ" in name:
            n = name.replace("XYZ", "")
            feature_names += [f"{n}ArCoeff-{ax}{i}" for ax in ["X", "Y", "Z"] for i in range(4)]
        else:
            feature_names += [f"{name}ArCoeff{i}" for i in range(4)]

    for name in [
        "tBodyAccXYZ",
        "tGravityAccXYZ",
        "tBodyAccJerkXYZ",
        "tBodyGyroXYZ",
        "tBodyGyroJerkXYZ",
    ]:
        n = name.replace("XYZ", "")
        feature_names += [f"{n}Correlation-{ax}" for ax in ["X", "Y", "Z"]]

    for name in freq_signal_names:
        for s in ["MaxInds", "MeanFreq", "Skewness", "Kurtosis"]:
            if "XYZ" in name:
                n = name.replace("XYZ", "")
                feature_names += [f"{n}{s}-{ax}" for ax in ["X", "Y", "Z"]]
            else:
                feature_names.append(f"{name}{s}")

    for name in ["tBodyAccXYZ", "tBodyAccJerkXYZ", "tBodyGyroXYZ"]:
        n = name.replace("XYZ", "")
        feature_names += [f"{n}BandsEnergy-{ax}{i}" for i in range(14) for ax in ["X", "Y", "Z"]]

    feature_names += [
        "tBodyAccWRTGravity",
        "tBodyAccJerkWRTGravity",
        "tBodyGyroWRTGravity",
        "tBodyGyroJerkWRTGravity",
        "tXAxisAccWRTGravity",
        "tYAxisAccWRTGravity",
        "tZAxisAccWRTGravity",
    ]

    feature_names += [
        f"tBody{sensor}ECDF-{axis}{i}"
        for sensor in ["Acc", "Gyro"]
        for axis in ["X", "Y", "Z"]
        for i in range(10)
    ]
    return feature_names
