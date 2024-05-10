import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

IMAGE_SIZE = (150, 150)

# Loading data
def load_data():
    DIRECTORY = r"E:\Projects\images"
    CATEGORY = ["seq_train", "seq_test"]

    images = []
    labels = []

    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        print("Loading {}".format(category))

        for folder in os.listdir(path):
            label = class_names_label[folder]

            # iterate through each image in the folder
            for file in os.listdir(os.path.join(path, folder)):

                # get the path of the image
                img_path = os.path.join(os.path.join(path, folder), file)

                # open and resize image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # append the image to corresponding label to the output
                images.append(image)
                labels.append(label)

    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='float32')

    return images, labels

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

def display_examples(class_names, images, labels):
    # display 25 images from the images array with corresponding labels
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Some of the images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = cv2.resize(images[i], IMAGE_SIZE)
        plt.imshow(image.astype(np.uint8))
        plt.xlabel(class_names[int(labels[i])])
    plt.show()

display_examples(class_names, train_images, train_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activaion= 'relu', input_shape = (150, 150, 3))
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.
])