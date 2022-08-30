---
layout: single
title: Vital signs monitoring using mm-wave radar with machine learning  
---

## Objectives
 - A non-contact health monitoring system to measure the heart rate, breathing rate and blood pressure using a frequency modulated continuous wave (FMCW) mm-wave radar.
 - Real-time prediction of arrhythmia (Cardiac disorder) in individuals using Deep learning.
 - Develop a novel non-contact blood pressure monitoring system using dual mm-wave FMCW radars using machine learning.

## Potential applications
 - Preliminary COVID screening : Real-time detection of COVID-like symptoms at supermarkets, malls, offices and other public places where this system can be used to detect HR, BR and BP in combination with temperature sensing cameras. 
 - Cardiac health monitoring and diagnostics: Vital signs can be visualized in real time on a dashboard for remote monitoring by health official and care takers. It can also be used as a diagnostic tool for prediction of disorders like arrhythmia, myocardial infarction, atrial fibrillation and other commonly found cardiac disorders in the elderly.
 - Real-time fitness conditioning of athletes and military personnel:  As a future use-case, object-tracking and localization can be integrated as a signal pre-processing step for monitoring BP for fitness and endurance conditioning of soldiers and athletes running on a treadmill. 
 - Prediction of random eye movement sleep behaviour disorders (RBD): Continuous monitoring of BP variability during night-time can be used to detect early onset of random eye movement sleep behaviour disorders (RBD) where a dreamer starts exhibiting uncontrollable behaviors usually accompanied by elevated heart, increased blood pressure and respiratory rates. 38-65% of RBD patients developed neurological disorders like Dementia, Parkinson’s disease, Multiple system atrophy and Alzheimer’s 7-13 years after the onset of RBD [1] which can be prevented if RBD is detected early on.

## Heart rate and breathing rate detection

 - The FMCW radar is placed pointed directly towards the chest of an individual some distance away. Each IF (intermediate frequency) signal obtained at the mixer is sampled at the beat frequency 𝑓_𝑏 and is converted to a complex range profile by applying the range FFT. 
 - Range profiles of multiple chirp signals are stacked on top of each other and converted into a matrix with i number of rows (Fast time Samples) and j columns (Slow time samples). Since, vital signs are detected for a stationary person, the phase change across the slow time axis is extracted from a single range bin. 
 - Phase unwrapping algorithm is then implemented to restrict the phase difference values to fall in [-𝜋,𝜋]. After filtering impulse noise due to phase unwrapping, the phase values are passed through a serially cascaded Bi-Quad IIR filter which filter out the signals based on a frequency range. The breathing wave is obtained in the frequency range 0.1-0.6Hz and heartbeat waves was extracted for 0.8-2Hz.
 - After the heartbeat waveform is placed in the buffer, additional signal processing steps included generation of QRS-like signals using Fourier series analysis. The QRS signals were constructed by extracting the peak amplitude and peak-peak interval of the original signal. 
 - 100,000 data samples are utilized to model each triangular wave component. For the input radar signal as shown above, n peaks are detected, hence there exists 100,000 x n data samples. Then the sampling frequency is set as 𝐹𝑠=(100000 𝑋 𝑛)/25.6𝑠  𝐻𝑧. The signal is then down-sampled to the desired sampling frequency Fs = 5Hz. 
 - The periodogram of the modelled signal as shown in Fig. 2 clearly shows that the PSD (power spectral density) lies at frequency = 1.133Hz which lies within the expected frequency range of a heartbeat phase signal i.e. 0.8- 2Hz. 

## Arrhythmia detection using deep learning

Preparation of the training dataset
The 3-layer ANN model is trained using 18 single lead ECG recordings from MIT-BIH normal sinus dataset and 15 readings from MIT-BIH arrhythmia dataset. 
Most commonly found ECG artifacts like Muscle tremor, electromagnetic interference (EMI) and base-line wander were filtered out: 
Muscle tremor caused due to shivering or sudden body movements (usually in the elderly) are high frequency signals 30~300Hz is removed by Butterworth low pass filters.
 The 50Hz Electromagnetic interference is removed by a Butterworth Band-stop filter.
Lastly, baseline wander is an ultra-low frequency signal that ranges between 0-0.8Hz that can be eliminated using a high-pass filter.
7 Training features extracted from the RR intervals (detected after filtering) include:

Preparation of testing dataset
As shown in Figure 3, the unseen test dataset is created from the heartbeat phase signals generated from the radar. 
Since the sampling frequency of the ECG signals used for training the deep learning model is 360Hz, the radar heartbeat signals were up-sampled from 2Hz to 360Hz. 
The RR interval based features stated above were then extracted from the heart beat phase signals.
Note: Since, there are no readily available datasets for radar based cardiac rhythm signals, the deep learning model was trained on ECG signals. 



