---
layout: single
title: Indoor/outdoor people counting and tracking using mmWave radar for privacy controlled surveillance!
---

## Objective 
To implement a people detection, tracking and counting system using TI's IWR1642BOOST EVM and raspberryPi

## Applications
 - Indoor/outdoor People counting and tracking for real-time surveillance with privacy control.
 - Motion detection
 - Automated doors/gates
 - Autonomous driving : Sensor fusion with cameras

## Overall working 

 - As shown in figure 1, the implementation of the people-counting application demo on the IWR1642 consists of a signal chain running on the C674x DSP, and the tracking module running on the ARM® Cortex®-R4F processor.
 - Raw ADC values are extracted, and a radar data cube is obtained. A radar cube is a 3-dimensional representation along the range, doppler and azimuth dimensions. 
The radar cube is first processed along its columns i.e. range using 1D FFT.
 - Next, to steer the beam vector and determine the Direction of arrival (DoA) of the received chirp accurately, the capon Beamforming (Minimum variance distortionless response) algorithm is used. It allows us to remove interference or noise due to external sources in the FOV.
 - Then, the CASO-CFAR (Cell average smallest of – Constant false alarm rate) algorithm is used for object detection based on a range-azimuth heatmap. To determine the velocity vector of objects detected, doppler FFT is computed.
 - Finally, after the point clouds of the objects are grouped and tracked, point cloud information is serially transmitted to the RPi through UART communication. For real time communication, a message broker( RabbitMQ) is used to upload a single observation to the cloud storage server.  

![]({{ site.baseurl }}/images/DPT_files/figure1.jpeg)

The above workflow can be subdivided into four main components:
 1. Range FFT through Range Azimuth Heatmap with Capon BF
 2. Object Detection with CFAR and Elevation Estimation
 3. Doppler Estimation
 4. Group Tracker

## 1.Range FFT through Range Azimuth Heatmap with Capon BF

As shown in figure 2, Raw Data is processed with a 1-D FFT (Range Processing) and Static Clutter Removal is applied to the result. Then Capon Beamforming is used to generate a range-azimuth heatmap. 

![]({{ site.baseurl }}/images/DPT_files/figure2.jpeg)

 1. Range processing
For each antenna, EDMA is used to move samples from the ADC output buffer to the FFT Hardware Accelerator (HWA), controlled by the Cortex R4F. A 16-bit, fixed-point 1D windowing and 16-bit, fixed-point, 1D FFT are performed. EDMA is used to move output from the HWA local memory to the radar cube storage in layer three (L3) memory. Range processing is interleaved with active chirp time of the frame. All other processing occurs each frame, except where noted, during the idle time between the active chirp time and the end of the frame.
 
 2. Static clutter removal
Once the active chirp time of the frame is complete, the interframe processing can begin, starting with static clutter removal. 1D FFT data is averaged across all chirps for a single Virtual Rx antenna. This average is then subtracted from each chirp from the Virtual Rx antenna. This cleanly removes the static information from the signal, leaving only the signals returned from moving objects. The formula is given below:

![]({{ site.baseurl }}/images/DPT_files/eqn1.png)
Where, Nc = Number of chirps; Nr = Number of recieve antennas; Xnr = Average samples for a single receive antenna across all chirps; Xncr = Samples of a Single Chirp from a receive antenna

 3. Capon beamforming
The Capon BF algorithm is split into two components: a) the Spatial Covariance Matrix computation and b) Range-Azimuth Heatmap Generation. The final output is the Range- Azimuth heatmap with beam weights. This is passed to the CFAR algorithm. 

  a) Spatial Covariance Matrix : The spatial covariance matrix is estimated as an average over the chirps in the frame as Rxx,n which is 8x8 for ISK and 4x4 for ODS:
                                                              ![]({{ site.baseurl }}/images/DPT_files/eqn2.png)
                                 Second, diagonal loading is applied to the R matrix to ensure stability :
                                                              ![]({{ site.baseurl }}/images/DPT_files/eqn3.png)

  b) Range-Azimuth Heatmap Generation: First, the Range-Azimuth Heatmap Pna is calculated using the following equations:
                                                   ![]({{ site.baseurl }}/images/DPT_files/eqn4.png)
                                       Then, the beamforming weights are calculated as :
                                                   ![]({{ site.baseurl }}/images/DPT_files/eqn5.png)

## 2. Object Detection with CFAR and Elevation Estimation

Using the heatmap generated in the range processing step, 2 Pass CFAR is used to generated detected points in the Range-Azimuth spectrum. For each detected point, Capon is applied to generate a 1D elevation angular spectrum, which is used to determine the elevation angle of the point.

![]({{ site.baseurl }}/images/DPT_files/figure3.jpeg)

 1. Object detection
Two pass CFAR algorithms is used on the range azimuth heat map to perform the object detection using the CFAR "smallest of" method. First pass is done per angle bin along the range domain. Second pass in the angle domain is used confirm the detection from the first pass. The output detected point list is stored in L2 memory.

 2. Elevation Estimation with Capon BF
Full 2D 12 antenna Capon Beamforming is performed at the azimuth of each detected point. This is done following the same steps used to generate the range-azimuth heatmap; a) generate spacial covariance matrix and b) generate 1D elevation angle spectrum (like the heatmap)

   a. Spatial Covariance matrix is like before, with input based on detections: 
                    ![]({{ site.baseurl }}/images/DPT_files/eqn6.png)
   b. 1D elevation angle spectrum: 
                    ![]({{ site.baseurl }}/images/DPT_files/eqn7.png)


## 3. Doppler Estimation 

For each detected point in range and azimuth(angle) space, Doppler is estimated using the capon beam weights and Doppler FFT. The output is stored in the L2 memory. This output is combined with the point cloud produced during CFAR and Elevation Estimation, resulting in output for each point of:
 1. Range
 2. Azimuth
 3. Elevation
 4. Doppler
 5. SNR

![]({{ site.baseurl }}/images/DPT_files/figure4.jpeg)


## 4. Group Tracker

It is possible to create multiple instances of group tracker. Figure 10 shows the steps algorithm goes during each frame call. The algorithm inputs measurement data in polar coordinates (range, azimuth, elevation, Doppler, SNR), and tracks objects in Cartesian space. Therefore, the extended Kalman filter (EKF) process is used.

![]({{ site.baseurl }}/images/DPT_files/figure5.jpeg)

Point cloud input is first tagged based on scene boundaries. Some points may be tagged as outside the boundaries and are ignored in association and allocation processes. Each iteration of the tracker runs through the following steps:
 1. If a track exists, use a Kalman filter to predict the tracking group centroid for time n based on state and process covariance matrices, estimated at time n-1. Compute a-priori state and error covariance estimations for each trackable object. At this step, compute measurement vector estimations.
 2. If a track exists, the association function allows each tracking unit to indicate whether each measurement point is close enough (gating), and if it is, to provide the bidding value (scoring). The point is assigned to a highest bidder.
 3. Any points not assigned to a track go through an allocate function. During the allocation process, points are first joined into a sets based on their proximity in 3D Cartesian + Doppler coordinates. 
 4. Each set becomes a candidate for an allocation decision and must pass a threshold for cumulative SNR and threshold for minimum number of points in the set to become a new track. When passed, the new tracking unit is allocated.
 5. During the update step, tracks are updated based on the set of associated points. Compute the innovation, Kalman gain, and a-posteriori state vector and error covariance. In addition to classic EKF, the error covariance calculation includes group dispersion in a measurement noise covariance matrix.
The report function queries each tracking unit and produces the algorithm output.





# Proof of concept 

### Features of the implemented system

![]({{ site.baseurl }}/images/DPT_files/figure6.jpeg)

### Expected outcomes of implemented system

 1. To be placed at entrances/exits to count number of people entering or leaving the facility. 
 2. To aid in counting people who are present in a zone/area.

### Setting up the IWR1642BOOST EVM

 - Voltage source: 5v (2.5A)
 - Tilt: 0o (Not required since we’re only detecting a person’s head/upper body)
 - Height: 1.5m
 - Power consumption: 2W
 - Data transfer: Micro USB for UART communication
 - Number of transmitting antennas:  2 (azimuth)
 - Number of receiving antennas: 4

### Calibration procedure for IWR1642

![]({{ site.baseurl }}/images/DPT_files/figure7.jpeg)

#### Step 1: set scene parameters

Modify the .cfg file as: SceneryParam -2 2 0.05 14 (Left right back front). This forms the area of interest in the form of a rectangle. People outside this zone will not be detected. To display the area of interest on the GUI, modify the distance to boundary parameters as shown in figure below.

![]({{ site.baseurl }}/images/DPT_files/figure8.jpeg)
















 




