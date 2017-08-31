# Vehicle Detection and Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_samples.jpg
[image2]: ./output_images/not_car_samples.jpg
[image3]: ./output_images/pipeline_test_window_search.jpg
[image4]: ./output_images/pipeline_test_image_tosearch.jpg
[image5]: ./output_images/pipeline_test_find_cars.jpg
[image6]: ./output_images/pipeline_test_heatmap.jpg
[image7]: ./output_images/car_hog_features.jpg
[image8]: ./output_images/not_car_hog_features.jpg
[video1]: ./project_video_output.mp4

---

## Histogram of Oriented Gradients (HOG)

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 98 through 115 of the file called `helper_functions.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

#### Vehicle images
![alt text][image1]

#### Non-vehicle images
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=13`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

#### Vehicle image
![alt text][image7]

#### Non-vehicle image
![alt text][image8]

### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found out that the following settings gave me a satisfied result when training the classifier:

```python
color_space = 'YCrCb'
orient = 13
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (16, 16)
hist_bins = 33
spatial_feat = True
hist_feat = True
hog_feat = True
```

### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using:

```
8792 car images
8968 not-car images
Using: 13 orientations, 8 pixels per cell and 2 cells per block
Feature vector length: 8511
23.57 Seconds to train SVC...
Test Accuracy of SVC = 0.9896
My SVC predicts:     [ 1.  1.  1.  1.  1.  0.  0.  1.  0.  1.]
For these 10 labels: [ 1.  1.  1.  1.  1.  0.  0.  1.  0.  1.]
0.00814 Seconds to predict 10 labels with SVC
```

I saved all parameter settings and the trained classifier into a pickle file called `parameters.p`. The training code can be found in the ```training()``` function starting at line 52 in `pipeline.py`.

## Sliding Window Search

### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I cropped the search area for the vehicles because approx. 50% - 55% of the upper half of the image are unnecessary for vehicle detection:
```python  
ystart = 400
ystop = 656
scale = 1.5
```

This is the search area for vehicles:
![alt text][image4]


### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched for vehicles converting the input images into the YCrCb color channel with 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result:
![alt text][image3]

---

## Video Implementation

### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the overlaid bounding boxes on the test image:
![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from the test image:
![alt text][image6]


---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?
I tested various parameters and picked the ones that were most suitable for my problem and gave me good results (98,96% test accuracy). I think if I was tweaking the parameters a little more I might have achieved a test accuracy over 99% but 98% is still good though (and the pipeline performs very well).
I think a more stable approach could be to save the last window positions (like in the advanced lane finding project) and predict where the next window will be (this might reduce the "wobblyness" of the bounding boxes). I also recognized that at some point in the video cars from the opposite track are detected as well and I think this might be a problem at some point too - like the detected false positives.

