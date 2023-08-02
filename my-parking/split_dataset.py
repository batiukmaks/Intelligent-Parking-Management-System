import os
import random
import shutil

# Set the paths to your image and label folders and the destination folders for splits
root_folder = "./annotated-data"
image_folder = "./annotated-data/raw-data"
label_folder = "./annotated-data/labels"
train_dest = "./dataset/train"
val_dest = "./dataset/valid"
test_dest = "./dataset/test"

subfolders = ["car-in", "car-out", "car-parked"]

# Set the percentage of data to use for training, valid, and testing
train_split = 0.7
val_split = 0.15
test_split = 0.15

for dir in [train_dest, val_dest, test_dest]:
    if not os.path.exists(dir):
        os.makedirs(os.path.join(dir, 'images'))
        os.makedirs(os.path.join(dir, 'labels'))

# Get the list of label files in the label folder
def get_label_files_recursively(label_folder, extension=".txt"):
    label_files = []
    for root, _, files in os.walk(label_folder):
        for file in files:
            if file.endswith(extension):
                label_files.append(os.path.join(root, file))
    return label_files

label_files = get_label_files_recursively(label_folder)

# Randomly shuffle the image files
random.shuffle(label_files)

# Calculate the number of images for each split
total_images = len(label_files)
train_size = int(train_split * total_images)
val_size = int(val_split * total_images)

# Split the dataset into training, valid, and testing sets
train_labels = label_files[:train_size]
val_labels = label_files[train_size:train_size + val_size]
test_labels = label_files[train_size + val_size:]

# Copy the images and corresponding label files to their respective splits
def copy_files(source_folder, destination_folder, label_paths):
    for label_path in label_paths:
        image_path = label_path.replace("labels", "raw-data")
        image_path = os.path.splitext(image_path)[0] + ".jpg"

        image_file = image_path.split('/')[-1]
        label_file = label_path.split('/')[-1]
        # print(image_file, label_file)
        # print(image_path, label_path)

        if not label_file == "classes.txt":
            shutil.copy(image_path, os.path.join(os.path.join(destination_folder, 'images'), image_file))
            shutil.copy(label_path, os.path.join(os.path.join(destination_folder, 'labels'), label_file))

copy_files(image_folder, train_dest, train_labels)
copy_files(image_folder, val_dest, val_labels)
copy_files(image_folder, test_dest, test_labels)
