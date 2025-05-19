# Kalman Tracking

This project is an example of Kalman Filters used for video object tracking.
Video object tracking is a process of dynamically creating a model for the location of an obejct within a video. It is similar to object detection. In both cases, the location of an object and the identity of an object are usually the most important problems.  


This demontrates video tracking using:
1. OpenCV Trackers: MOSSE
2. Kalman filter and smoothing methods

# How to run

* Download [BIRSDAI](https://sites.google.com/view/elizabethbondi/dataset) dataset
* Download [TRICLOBS](https://figshare.com/articles/dataset/The_TRICLOBS_Dynamic_Multiband_Image_Dataset/3206887) dataset 

* Configure filepaths
* Run tracking_kalman_filter.py

### Prerequisites

- Create a virtual environment:
     ```bash
     python3 -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows: `venv\Scripts\activate`
     - On Unix or MacOS: `source venv/bin/activate`

- Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

### Running the Application

1. **MOSSE Tracker:**
  - You can run the MOSSE tracker using:
    ```
    python src/main.py
    ```
  - This will also install the dataset automatically if it's not already present.
  - You can also run it on your webcam using: ```python src/main_webcam.py```
    
2. **Kalman Filter Tracking:**
  - To run the Kalman filter tracking, execute:
    ```bash
    python tracking_kalman_filter.py
    ```

    
# Scenarios
Looking at two different scenarios using two different datasets. 

The first dataset comes from a wildlife drone dataset, BIRDSAI [[1]](https://sites.google.com/view/elizabethbondi/dataset), containing thermal infrared scenes with elephants, giraffes, humans, and other animals that must be tracked. In the image below on hte left, wildlife is being monitored with a drone using a TIR camera. Two elephants are detected, but with noisy data and the motion of the drone, some detections are masked over time. 

The second dataset used is TRICLOBS [[2]](https://figshare.com/articles/dataset/The_TRICLOBS_Dynamic_Multiband_Image_Dataset/3206887), a tri-band infrared video which contains surveillance scenes of humans in military contexts. In the image below on the right, humans are being monitored in a town. There are three people in the car, but it is difficult to detect at first. The humans leave the car and walk out of the scene.

* Download [BIRSDAI](https://sites.google.com/view/elizabethbondi/dataset) dataset
* Download [TRICLOBS](https://figshare.com/articles/dataset/The_TRICLOBS_Dynamic_Multiband_Image_Dataset/3206887) dataset 

**Example images from the BIRSDAI (left) and TRICLOBS (right) scenes**:


![image](https://github.com/bradleeharr/MultiBandIRTracking/assets/56418392/18560f3f-ac92-4a85-a0fc-18ef5c30dd39)
![image](https://github.com/bradleeharr/MultiBandIRTracking/assets/56418392/1959ce23-2f55-4bb9-a027-724e6701a53d)

# Results

An example result from tracking an elephant in scene 1 of the BIRSDAI dataset
<p align="center">

Using a constant-velocity model:
</p>

<p align="center"> 
<img src="https://github.com/bradleeharr/MultiBandIRTracking/assets/56418392/a2ae3367-0f2f-4b62-ac76-2817901bb3ee"/>
</p>
<p align="center">

Using a constant-acceleration model:
</p>

<p align="center">
<img src="https://github.com/bradleeharr/MultiBandIRTracking/assets/56418392/5f83389b-5261-4754-b373-377a29256a43"/>
</p>

In this case, the constant-velocity model is far superior than the constant acceleration model, and fluctuations due to noise cause significant error in the constant acceleration model. 
The drone used to record the footage moves with constant velocity, so it is reasonable that the constant velocity scenario accurately models the motion.


