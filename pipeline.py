import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from helper_functions import *


def create_pickle(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat,
                  hist_feat, hog_feat, X_scaler, svc):
    params = {"color_space": color_space, "orient": orient, "pix_per_cell": pix_per_cell,
              "cell_per_block": cell_per_block, "hog_channel": hog_channel, "spatial_size": spatial_size,
              "hist_bins": hist_bins, "spatial_feat": spatial_feat, "hist_feat": hist_feat, "hog_feat": hog_feat,
              "scaler": X_scaler, "svc": svc}
    pickle.dump(params, open("parameters.p", "wb"))


def save_random_images(imgs, filename, amount=10, hog_features=False):
    fig, axs = plt.subplots(1, amount, figsize=(20, 10))

    if hog_features:
        rand_img = imgs[np.random.randint(0, len(imgs))]
        hog_images = [rand_img]
        titles = ["Original"]
        for i in range(3):
            features, hog_img = get_hog_features(rand_img[:, :, i], orient=13, pix_per_cell=8, cell_per_block=2,
                                                 vis=True)
            hog_images.append(hog_img)
            titles.append("HOG Channel {}".format(i))

    for i in range(amount):
        if hog_features:
            rand_img = hog_images[i]
            axs[i].set_title(titles[i])
            axs[i].imshow(rand_img, cmap='gray')
        else:
            rand_img = imgs[np.random.randint(0, len(imgs))]
            axs[i].imshow(rand_img)
        axs[i].axis("off")
        axs[i].tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off',
                           labeltop='off',
                           labelright='off', labelbottom='off')
    plt.savefig("output_images/" + filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def training():
    # set parameters
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 13  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 33  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    not_car_features = extract_features(not_cars, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using: {} orientations, {} pixels per cell and {} cells per block'.format(orient, pix_per_cell,
                                                                                     cell_per_block))
    print('Feature vector length: {}'.format(len(X_train[0])))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print('{} Seconds to train SVC...'.format(round(t2 - t, 2)))
    # Check the score of the SVC
    print('Test Accuracy of SVC = {}'.format(round(svc.score(X_test, y_test), 4)))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts:     {}'.format(svc.predict(X_test[0:n_predict])))
    print('For these {} labels: {}'.format(n_predict, y_test[0:n_predict]))
    t2 = time.time()
    print('{} Seconds to predict {} labels with SVC'.format(round(t2 - t, 5), n_predict))

    # save params in a pickle file
    create_pickle(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat,
                  hist_feat, hog_feat, X_scaler, svc)


def video_pipeline(img, save=False):
    if not Path("parameters.p").is_file():
        training()
    params_pickle = pickle.load(open('parameters.p', 'rb'))
    color_space = params_pickle["color_space"]
    orient = params_pickle["orient"]
    pix_per_cell = params_pickle["pix_per_cell"]
    cell_per_block = params_pickle["cell_per_block"]
    hog_channel = params_pickle["hog_channel"]
    spatial_size = params_pickle["spatial_size"]
    hist_bins = params_pickle["hist_bins"]
    spatial_feat = params_pickle["spatial_feat"]
    hist_feat = params_pickle["hist_feat"]
    hog_feat = params_pickle["hog_feat"]
    X_scaler = params_pickle["scaler"]
    svc = params_pickle["svc"]

    if save:
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        # Test images are always in .jpg!
        jpeg_img = img.astype(np.float32) / 255

        draw_image = np.copy(jpeg_img)
        # Min and max in y to search in slide_window()
        # assuming that approx. 55% of the upper half of the image are unnecessary
        y_start_stop = [int(draw_image.shape[0] * 0.55), None]
        windows = slide_window(jpeg_img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                               xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        hot_windows = search_windows(jpeg_img, windows, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        mpimg.imsave("output_images/pipeline_test_window_search.jpg", window_img)

    ystart = 400
    ystop = 656
    scale = 1.5
    if save:
        img_tosearch = img[ystart:ystop, :, :]
        mpimg.imsave("output_images/pipeline_test_image_tosearch.jpg", img_tosearch)

    out_img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size, hist_bins)
    if save:
        mpimg.imsave("output_images/pipeline_test_find_cars.jpg", out_img)

    result, heatmap_img = get_heatmap_img(img, box_list)

    if save:
        fig = plt.figure()
        plt.axis('off')
        plt.subplot(121)
        plt.imshow(result)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap_img, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        fig.savefig("output_images/pipeline_test_heatmap.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

    return result


# read in not-car and car images
not_cars = read_in_images('non-vehicles/*/*.png')
cars = read_in_images('vehicles/*/*.png')

print("{} car images".format(len(cars)))
print("{} not-car images".format(len(not_cars)))

# save 10 random example images
save_random_images(not_cars, filename="not_car_samples.jpg")
save_random_images(cars, filename="car_samples.jpg")

# save 1 random example image with all hog feature visualization in all 3 color channels
save_random_images(not_cars, filename="not_car_hog_features.jpg", amount=4, hog_features=True)
save_random_images(cars, filename="car_hog_features.jpg", amount=4, hog_features=True)

# train the classifier
training()

# take a test image and test the pipline
img = mpimg.imread('test_images/test1.jpg')
video_pipeline(img, save=True)

# try the pipline on a videotream
video_output_file = 'project_video_output.mp4'
video_input_file = VideoFileClip("./project_video.mp4")
video_output = video_input_file.fl_image(video_pipeline)
video_output.write_videofile(video_output_file, audio=False)
