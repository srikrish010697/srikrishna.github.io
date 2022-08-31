---
layout: single
title: NN_XGB stacked ensemble 
---

## 1.	INTRODUCTION
Quality measurement is done during production to ensure optimal machine operation and reduce defective units. Existing quality analysis is restricted to a random sampling of bonded units using manual or automatic metrology equipment, some of which are destructive tests that lead to wastage. Hence, we propose an AI-based predictive metrology method to predict quality measurement for all units in production in near-real-time. As an equipment manufacturer, we have access to multiple sensor data collected from wire bonding machines in production useful for predictive metrology. We explore how we can utilise this data for quality prediction modelling.

## 2.	METHODOLOGY
I designed and implemented an end-to-end ML pipeline for quality prediction with the following components:
1.	Data pre-processing: Data pre-processing refers to data transformation steps carried out to ensure data quality. These transformations handle NUL lines, trailing commas, extra/missing columns, invalid column type, duplicated/missing data, and data out of range.
2.	Feature engineering: Based on our domain expertise, we extracted multiple process and statistical-based features derived from raw time series signals extracted from machine sensors. The signals detect features related to heat, pressure, and ultrasonic vibration.
3.	Model training: The trained model predicts continuous valued non-destructive quality measurement by learning non-linear relationships between input features and quality metrology ground truth. Two ensemble stacking models are introduced namely, Extra Trees and NN_XGB ensemble which are compared to state-of-the-art algorithms that include XGBoost, GradientBoost, Neural networks, AdaBoost and ExtraTrees. 
4.	Model optimisation: To ensure model hyperparameters were tuned correctly, multiple optimisation algorithms were explored to select the best one that minimises the loss function. Algorithms include Genetic Algorithm [1] and Bayesian Optimisation [2].
5.	Model retraining methodology: To handle input data drifts and improve model performance over longer periods of production data, online training approach was used to retrain the models.
6.	Model performance dashboard: A dashboarding tool was implemented to monitor the quality prediction statistics and model performance.

## 3.	RESULTS
The proposed architecture for ensemble learning as described in Figure 1, uses a stacked ensemble of base learners and a XGBoost meta model. The base learner hyperparameters are tuned using training data with 5-fold cross validation using Bayesian and Genetic optimisation algorithms. N base model predictions are then used to train the XGBoost meta learner, where the target is actual QA measurement in training set. For extra trees’ ensemble, two extra tree models serve as base learners while NN_XGB uses two neural network, 2 XGBoost and 1 Extra Tree models as base learners. 

![Figure 1: Proposed ensemble stacked model]({{ site.baseurl }}/images/DPT_files/ensemble/Picture1.png))

 
The following regression algorithms are compared based on an unseen test dataset. From Table 1, we found that the proposed extra trees ensemble model outperformed all the other algorithms including extra trees based on aggregated score, NRMSE, NMAE and R-squared value. The aggregated score (higher is better) accounts for the model’s bias-variance trade-off. The NN_XGB model performed better than neural network and AdaBoost models, and nearly as good as the boosting algorithms namely XGBoost and Gradient Boost. Hence, the proposed ensemble methods are more robust and generalise well for an unseen test dataset than their baseline counterparts. 

![]({{ site.baseurl }}/images/DPT_files/ensemble/Picture2.png))


## 4.	NOVELTY and IMPACT
 - Based on the test performance, we achieved better performance using the proposed ensemble models compared to their baseline methods based on prediction error. 
 - Our predictive metrology method can predict quality measurement on 100% of units produced without destructive tests which saves cost and man hours.
 - Additionally, this method can be used to raise failure alerts during production whenever quality degradation is predicted. Using Explainable AI [3] algorithms, 
we can flag out important process-related features that can assist operators/technicians to optimize production process parameters, thereby minimising yield loss. 
 - This quality failure diagnostic tool can help domain experts troubleshoot sub-optimal machines and uncover relationships between machine processes and bond quality. 
 - The proposed usage of online learning mechanism can detect data distribution drifts due to changing bonding conditions, software updates, and other calibration operations. 
 - This framework can automatically update model hyperparameters during live production, ensuring that predictive metrology services are not disrupted. 
 - Hence, minimising compute costs, training time and elimination of manual retraining by data scientists. 

## REFERENCES
[1]	Katoch, S., Chauhan, S.S. & Kumar, V. A review on genetic algorithm: past, present, and future. Multimed Tools Appl 80, 8091–8126 (2021). https://doi.org/10.1007/s11042-020-10139-6

[2]	B. Shahriari, K. Swersky, Z. Wang, R. P. Adams and N. de Freitas, "Taking the Human Out of the Loop: A Review of Bayesian Optimization," in Proceedings of the IEEE, vol. 104, no. 1, pp. 148-175, Jan. 2016, doi: 10.1109/JPROC.2015.2494218.

[3]	Vilone, Giulia, and Luca Longo. "Explainable artificial intelligence: a systematic review." arXiv preprint arXiv:2006.00093 (2020).
