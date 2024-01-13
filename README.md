# Kalman Tracking
Video tracking using kalman filter and smoothing methods

# How to run

* Download [BIRSDAI](https://sites.google.com/view/elizabethbondi/dataset) dataset
* Download [TRICLOBS](https://figshare.com/articles/dataset/The_TRICLOBS_Dynamic_Multiband_Image_Dataset/3206887) dataset 

* Configure filepaths
* Run tracking_kalman_filter.py

# Scenarios
I looked at two different scenarios using two different datasets. 

The first dataset comes from a wildlife drone dataset, BIRDSAI [[1]](https://sites.google.com/view/elizabethbondi/dataset), containing thermal infrared scenes with elephants, giraffes, humans, and other animals that must be tracked. In the image below on hte left, wildlife is being monitored with a drone using a TIR camera. Two elephants are detected, but with noisy data and the motion of the drone, some detections are masked over time. 

The second dataset used is TRICLOBS [[2]](https://figshare.com/articles/dataset/The_TRICLOBS_Dynamic_Multiband_Image_Dataset/3206887), a tri-band infrared video which contains surveillance scenes of humans in military contexts. In the image below on the right, humans are being monitored in a town. There are three people in the car, but it is difficult to detect at first. The humans leave the car and walk out of the scene.

**Example images from the BIRSDAI (left) and TRICLOBS (right) scenes**:


![image](https://github.com/bradleeharr/MultiBandIRTracking/assets/56418392/18560f3f-ac92-4a85-a0fc-18ef5c30dd39)
![image](https://github.com/bradleeharr/MultiBandIRTracking/assets/56418392/1959ce23-2f55-4bb9-a027-724e6701a53d)

# Results

Some example results from tracking an elephant in scene 1 of the BIRSDAI dataset

Using a constant-velocity model:

![image](https://github.com/bradleeharr/MultiBandIRTracking/assets/56418392/a2ae3367-0f2f-4b62-ac76-2817901bb3ee)

Using a constant-acceleration model:

![image](https://github.com/bradleeharr/MultiBandIRTracking/assets/56418392/5f83389b-5261-4754-b373-377a29256a43)

We see that the constant-velocity model is far superior than the constant acceleration model, and fluctuations due to noise cause significant error in the constant acceleration model. 
The drone used to record the footage moves with constant velocity, so it is reasonable that the constant velocity scenario accurately models the motion.


