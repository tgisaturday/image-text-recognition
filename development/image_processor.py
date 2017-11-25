import numpy as np
import cv2

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from skimage.transform import resize

from matplotlib import pyplot as plt
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage import restoration
from skimage import measure
from skimage.color import label2rgb
import matplotlib.patches as mpatches


import tensorflow as tf
import cnn_predictor_all as cnna
import cnn_predictor_digit as cnnd



class ImageProcess():
    def __init__(self, image_file, pre_flag):
        tmp_img = cv2.imread(image_file, 0)
        #tmp_img = cv2.medianBlur(tmp_img, 3)
        #tmp_img = cv2.GaussianBlur(tmp_img, (5, 5), 0)
        tmp_img = cv2.bitwise_not(tmp_img)
        self.image = np.asarray(tmp_img)
        self.bw = self.image
        self.cleared = self.bw.copy()
        if pre_flag == True:
            self.preprocess()

    def preprocess(self):
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        #thresh = threshold_local(image, block_size, offset=10)
        thresh = threshold_otsu(image)
        self.bw = closing(image > thresh, square(2))
        self.cleared = self.bw.copy()
        return self.cleared

    def get_candidates(self):
        """
        identifies objects in the image. Gets contours, draws rectangles around them
        and saves the rectangles as individual images.
        """
        label_image = measure.label(self.cleared)
        borders = np.logical_xor(self.bw, self.cleared)
        label_image[borders] = -1
        coordinates = []
        i = 0

        for region in regionprops(label_image):
            if region.area > 10:
                minr, minc, maxr, maxc = region.bbox
                margin = 3
                minr, minc, maxr, maxc = minr - margin, minc - margin, maxr + margin, maxc + margin
                roi = self.image[minr:maxr, minc:maxc]
                if roi.shape[0] * roi.shape[1] == 0:
                    continue
                else:
                    width = maxc - minc;
                    height = maxr - minr;
                    if width > height:
                        new_width = 22
                        new_height = round(22 * height / width)
                        if new_height == 0:
                            new_height = 2
                        if new_height % 2 != 0:
                            new_height += 1

                    elif width < height:
                        new_width = round(22 * width / height)
                        new_height = 22
                        if new_width == 0:
                            new_width = 2
                        if new_width % 2 != 0:
                            new_width += 1
                    else:
                        new_width = 22
                        new_height = 22

                    if i == 0:
                        #resizing individual image into 12 x 20 pixel image
                        #This is because usual font images have longer height than width
                        samples = resize(roi, (new_height,new_width), mode='constant')
                        #padding extra pixel to zero in order to fit it in 28x28 convolution layer
                        samples = np.pad(samples, ((int((28-new_height)/2), int((28-new_height)/2)),
                                                   (int((28-new_width)/2), int((28-new_width)/2))), mode='constant')
                        coordinates.append(region.bbox)
                        i += 1
                    elif i == 1:
                        roismall = resize(roi, (new_height, new_width), mode='constant')
                        roismall = np.pad(roismall, ((int((28-new_height)/2), int((28-new_height)/2)),
                                                     (int((28-new_width)/2), int((28-new_width)/2))), mode='constant')
                        samples = np.concatenate((samples[None, :, :], roismall[None, :, :]), axis=0)
                        coordinates.append(region.bbox)
                        i += 1
                    else:
                        roismall = resize(roi, (new_height, new_width), mode='constant')
                        roismall = np.pad(roismall, ((int((28-new_height)/2), int((28-new_height)/2)),
                                                     (int((28-new_width)/2), int((28-new_width)/2))), mode='constant')
                        samples = np.concatenate((samples[:, :, :], roismall[None, :, :]), axis=0)
                        coordinates.append(region.bbox)

        self.candidates = {
            'fullscale': samples,
            'flattened': samples.reshape((samples.shape[0], -1)),
            'coordinates': np.array(coordinates)
        }

        return self.candidates

    def plot_preprocessed_image(self):
        """
        plots pre-processed image. The plotted image is the same as obtained at the end
        of the get_text_candidates method.
        """
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(2))
        cleared = bw.copy()

        label_image = measure.label(cleared)
        borders = np.logical_xor(bw, cleared)

        label_image[borders] = -1
        image_label_overlay = label2rgb(label_image, image=image)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        ax.imshow(image_label_overlay)

        for region in regionprops(label_image):
            if region.area < 10:
                continue

            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        plt.show(block=False)


    def plot_to_check(self, what_to_plot, title):
        """
        plots images at several steps of the whole pipeline, just to check output.
        what_to_plot is the name of the dictionary to be plotted
        """
        n_images = what_to_plot['fullscale'].shape[0]

        fig = plt.figure(figsize=(12, 12))

        if n_images <= 100:
            if n_images < 100:
                total = range(n_images)
            elif n_images == 100:
                total = range(100)

            for i in total:
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][i], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, "{}".format(str(what_to_plot['predicted_char'][i])), fontsize=22, color='red')
                    #ax.text(-6, 8, "{}: {:.2}".format(str(what_to_plot['predicted_char'][i]),
                            #float(max(what_to_plot['predicted_prob'][i].tolist()))), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show(block=False)

        else:
            total = list(np.random.choice(n_images, 100))
            for i, j in enumerate(total):
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['fullscale'][j], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, "{}".format(str(what_to_plot['predicted_char'][i])), fontsize=22, color='red')
                    #ax.text(-6, 8, "{}: {:.2}".format(str(what_to_plot['predicted_char'][i]),
                            #float(max(what_to_plot['predicted_prob'][i].tolist()))), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show(block=False)


    def predict_char(self):
        """
        Using 5 ConvNet Models to predict each character.
        With means of softmax from 5 networks and using argmax, predicts closest character.
        """

        softmax = []

        g1 = tf.Graph()
        g2 = tf.Graph()
        g3 = tf.Graph()
        g4 = tf.Graph()
        g5 = tf.Graph()
        g6 = tf.Graph()
        g7 = tf.Graph()
        g8 = tf.Graph()
        g9 = tf.Graph()

        with g1.as_default():
            softmax_tmp = cnnd.char_prediction(self.candidates['flattened'], 1)
            softmax.append(softmax_tmp)

        with g2.as_default():
            softmax_tmp = cnnd.char_prediction(self.candidates['flattened'], 2)
            softmax.append(softmax_tmp)

        with g3.as_default():
            softmax_tmp = cnnd.char_prediction(self.candidates['flattened'], 3)
            softmax.append(softmax_tmp)

        with g4.as_default():
            softmax_tmp = cnnd.char_prediction(self.candidates['flattened'], 4)
            softmax.append(softmax_tmp)


        with g5.as_default():
            softmax_tmp = cnnd.char_prediction(self.candidates['flattened'], 5)
            softmax.append(softmax_tmp)

        with g6.as_default():
            softmax_tmp = cnnd.char_prediction(self.candidates['flattened'], 6)
            softmax.append(softmax_tmp)

        with g7.as_default():
            softmax_tmp = cnna.char_prediction(self.candidates['flattened'], 7)
            softmax.append(softmax_tmp)

        with g8.as_default():
            softmax_tmp = cnna.char_prediction(self.candidates['flattened'], 8)
            softmax.append(softmax_tmp)

        with g9.as_default():
            softmax_tmp = cnna.char_prediction(self.candidates['flattened'], 9)
            softmax.append(softmax_tmp)


        predicted_softmax_digit = softmax[0]
        for i in range(1, 6):
            predicted_softmax_digit += softmax[i]
        predicted_softmax_digit /= 6
        predicted_argmax_digit = tf.argmax(predicted_softmax_digit, 1).eval(session=tf.Session())

        predicted_softmax_all = (softmax[6] + softmax[7] + softmax[8])/3
        predicted_argmax_all = tf.argmax(predicted_softmax_all, 1).eval(session=tf.Session())

        self.which_text = {
                                 'fullscale': self.candidates['fullscale'],
                                 'flattened': self.candidates['flattened'],
                                 'coordinates': self.candidates['coordinates'],
                                 'predicted_digit': predicted_argmax_digit,
                                 'predicted_prob_digit': predicted_softmax_digit,
                                 'predicted_all': predicted_argmax_all,
                                 'predicted_prob_all': predicted_softmax_all
                                 }
        return self.which_text

    def realign_text(self):
        """
        processes the classified characters and reorders them in a 2D space
        generating a matplotlib image.
        """
        max_maxrow = max(self.which_text['coordinates'][:, 2])
        min_mincol = min(self.which_text['coordinates'][:, 1])
        subtract_max = np.array([max_maxrow, min_mincol, max_maxrow, min_mincol])
        flip_coord = np.array([-1, 1, -1, 1])

        coordinates = (self.which_text['coordinates'] - subtract_max) * flip_coord

        ymax = max(coordinates[:, 0])
        xmax = max(coordinates[:, 3])

        predicted_digit = self.which_text['predicted_digit']
        predicted_prob_digit = self.which_text['predicted_prob_digit']
        predicted_prob_digit = [max(list(predicted_prob_digit)) for predicted_prob_digit in predicted_prob_digit]

        predicted_all = self.which_text['predicted_all']
        predicted_prob_all = self.which_text['predicted_prob_all']
        predicted_prob_all = [max(list(predicted_prob_all)) for predicted_prob_all in predicted_prob_all]

        coordinates = [list(coordinate) for coordinate in coordinates]

        predicted_value = []
        predicted_prob = []
        for i in range(0, len(predicted_digit)):
            if predicted_prob_all[i] > 0.5:
                if predicted_all[i] > 10:
                    if predicted_all[i] < 36:
                        temp = chr(ord('a')+predicted_all[i]-10)
                    else:
                        temp = chr(ord('a')+predicted_all[i]-36)
                    predicted_value.append(temp)
                    predicted_prob.append(predicted_prob_all[i])
                else:
                    predicted_value.append(str(predicted_digit[i]))
                    predicted_prob.append(predicted_prob_digit[i])
            else:
                predicted_value.append(str(predicted_digit[i]))
                predicted_prob.append(predicted_prob_digit[i])


        #solves python3 zip() problem
        realign_tmp = list(zip(coordinates, predicted_value, predicted_prob))
        to_realign_tmp = realign_tmp[:]
        to_realign = list(to_realign_tmp)

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        for char in to_realign:

            if(char[2] <0.5):
                pass
                ax.text(char[0][1], char[0][2], char[1], size=16, color='black')
            elif (char[2] <0.70):
                ax.text(char[0][1], char[0][2], char[1], size=16, color='black')
            elif(char[2] <0.90):
                ax.text(char[0][1], char[0][2], char[1], size=16, color='red')
            elif (char[2] < 0.95):
                ax.text(char[0][1], char[0][2], char[1], size=16, color='blue')
            else:
                ax.text(char[0][1], char[0][2], char[1], size=16, color='green')

        ax.set_ylim(-10, ymax + 10)
        ax.set_xlim(-10, xmax + 10)

        plt.show(block=False)
        return to_realign
