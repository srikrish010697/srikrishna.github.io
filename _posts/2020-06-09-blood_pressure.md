---
layout: single
title: Non-Contact Blood Pressure Monitoring System using Dual mm-Wave Radar sensors
---

## Working principal 

 - Phase signals are extracted from two mm-wave radars A & B pointed toward the chest and neck as shown in Figure 1
 - The radar B pointed towards the neck region measures the tiny displacements of the carotid artery due to blood flow while radar A pointed towards the chest, measures the displacements due to heartbeat and breathing.  
 - After the required neck pulse waveforms and heartbeat signals are extracted, peak detection is performed to extract PTT-based features. 
 - PTT (Pulse transit time) is defined as the time it takes for a pulse pressure signal to flow between two points on the arterial network. Since there exists a non-linear relationship between PTT and BP, a ECG-PPG signal dataset is used to train a machine learning model to predict SBP and DBP values. 



## Preparation of the training dataset

 - A non-invasive blood presure dataset from kaggle was used to train a regression model. 
 - The dataset consists of PPG, ECG, FSR (Force sensing resistor), and PCG (Phonocardiogram) signals with the corresponding  SBP and DBP values. The dataset was formed using 26 test subjects in the age range, 21-50 years.  
 - Frequency of a PPG signal normally lies in the range 0.5-5Hz while for motion artifacts it lies in 0.01-10Hz. the moving median filter is used to remove motion artifact since it has the least computation time. 
 - The PTT is measured by computing the time period between the onset of the R wave in an ECG signals and the PPG peak amplitude signal. 
