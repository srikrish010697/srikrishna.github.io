---
layout: single
title: Non-Contact Blood Pressure Monitoring System using Dual mm-Wave Radar sensors
---

## Working principal 

![]({{ site.baseurl }}/images/DPT_files/vital_signs/figure9.png)

 - Phase signals are extracted from two mm-wave radars A & B pointed toward the chest and neck as shown in figure below
 - The radar B pointed towards the neck region measures the tiny displacements of the carotid artery due to blood flow while radar A pointed towards the chest, measures the displacements due to heartbeat and breathing.  
 - After the required neck pulse waveforms and heartbeat signals are extracted, peak detection is performed to extract PTT-based features. 
 - PTT (Pulse transit time) is defined as the time it takes for a pulse pressure signal to flow between two points on the arterial network. Since there exists a non-linear relationship between PTT and BP, a ECG-PPG signal dataset is used to train a machine learning model to predict SBP and DBP values. 

![]({{ site.baseurl }}/images/DPT_files/vital_signs/figure7.png)

## Preparation of the training dataset

 - A non-invasive blood presure dataset from kaggle was used to train a regression model. 
 - The dataset consists of PPG, ECG, FSR (Force sensing resistor), and PCG (Phonocardiogram) signals with the corresponding  SBP and DBP values. The dataset was formed using 26 test subjects in the age range, 21-50 years.  
 - Frequency of a PPG signal normally lies in the range 0.5-5Hz while for motion artifacts it lies in 0.01-10Hz. the moving median filter is used to remove motion artifact since it has the least computation time. 
 - The PTT is measured by computing the time period between the onset of the R wave in an ECG signals and the PPG peak amplitude signal. 

![]({{ site.baseurl }}/images/DPT_files/vital_signs/figure8.png)

## Comparing training algorithms

 - The diastolic blood pressure (DBP) database was trained with 14 regression algorithms and compared based on minimization of RMSE (root mean square error), R2 (regression coefficient), MSE (mean square error), MAE (mean absolute error) and training time, in seconds. The algorithms were cross validated using 3-fold cross-validation.
 - From Table 1 and 2, Gaussian SVM clearly outperforms other algorithms based on minimization of RMSE & MAE and maximization of the regression coefficient.

![]({{ site.baseurl }}/images/DPT_files/vital_signs/figure10.png)

## Hyperparameter tuning

 - Tunable parameters include Kernel scale,  box constraint , epsilon value (tolerance level) , Number of iterations
 - The minimum MSE plot in Fig. 10 shows the gradually decreasing MSE for every iteration number. 
 - The evaluation results are compared based on three optimization techniques as outlined in Table 5 and 6 . Since Bayesian optimization yields a trained model with RMSE ≈ 0 and R2=1, it can be inferred that the model is overfit and cannot be utilized to predict SBP values. 
 - Both Bayesian and Grid search optimization fail to converge to the local minima while random search produced better results with reduced RMSE value from 2.058 to 1.495 and an improvement in the R2 value from 0.95 to 0.97. 
 - Hence random search optimization algorithm is employed for both SBP and DBP prediction.

![]({{ site.baseurl }}/images/DPT_files/vital_signs/figure11.png)

![]({{ site.baseurl }}/images/DPT_files/vital_signs/figure12.png)

## Testing results

 - The Radar system is tested at varying distances and BP values are recorded at d=30cm, 60 cm, 90 cm, 120 cm & 150 cm as shown in Table VI to determine the ideal range for accurate BP measurement. 
 - For every distance measured, (d=30 cm, 60 cm, 90 cm, 120 cm & 150 cm) 10 observations were recorded for a single test subject. 
 - Each radar observation lasted for 25.6s while that of OMRON takes approximately 60s to obtain the BP outputs. For every observation recorded by the radar system, a corresponding SBP & DBP value is recorded by OMRON. 
 - The observations were then evaluated based on standard deviation, p statistic, correlation coefficient, RMSE value, MAE and MedAE. A comparison was made between ideal (OMRON) and radar-based measurements. 
 - From Table 7, both Range = 120 cm and Range =150 cm produce uncertain diagnostic results. For 150 cm, both Hypertension stage-I and normal BP are tied at 4 observations each. Based on this deviation in the diagnostic results, we can conclude that 150 cm is unsuitable for measurement of our 77 GHz radar system. 
 - Additionally, we also observe uncertainty in diagnosis for 120 cm as 5 out of 10 observations recorded by the radar are normal while the remaining 5 observations show that they are elevated BP values.

![]({{ site.baseurl }}/images/DPT_files/vital_signs/figure13.png)

 - Our experimental results show that distances 30 cm, 60 cm, 90 cm and 120 cm are well within the AAMI standards for non-invasive BP measurement. 
 - To further support this claim, the p-values and R can be analysed for 10 observations. 
 - Consider a null hypothesis which states that there is simply no relation between the radar-based and OMRON-based SBP & DBP values. If the p value obtained is <0.05, then we can safely conclude that the probability of accepting the null hypothesis is low i.e. we can safely reject the null hypothesis. 
 - For example, in Table 8, the p-value for SBP is 0.004 which implies that there is a probability of 0.004 of accepting the null hypothesis. Since this probability is too low, we can safely reject the null hypothesis thereby inferring that there exists a linear relationship between the radar-based and OMRON-based SBP values. 
 - The maximum R value for SBP and DBP measurement is obtained when the Range is 30 cm and 90 cm respectively
 - As expected, for 30 cm-120 cm, a positive correlation coefficient is obtained but a negative R value of -0.73 and -0.44 is found at 150 cm. This proves that there exists a negative correlation between the BP values estimated by radar and OMRON at 150 cm thereby justifying the rejection of the null hypothesis previously considered. 
 - Hence, it can be concluded that the 30 cm-90 cm range is ideal for BP measurement using the proposed system. 

![]({{ site.baseurl }}/images/DPT_files/vital_signs/figure14.png)




For more information, please refer to []


