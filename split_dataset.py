import os
import random
import shutil

# Set the paths to your image and label folders and the destination folders for splits
root_folder = "./annotated-data"
image_folder = "./annotated-data/raw-data"
label_folder = "./annotated-data/labels"
train_dest = "./default-dataset/train"
val_dest = "./default-dataset/valid"
test_dest = "./default-dataset/test"

subfolders = ["car-in", "car-out", "car-parked"]

# Set the percentage of data to use for training, valid, and testing
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Create destination folders for training, validation, and testing splits
for dir in [train_dest, val_dest, test_dest]:
    if not os.path.exists(dir):
        os.makedirs(os.path.join(dir, 'images'))
        os.makedirs(os.path.join(dir, 'labels'))

# Function to get the list of label files in the label folder recursively
def get_label_files_recursively(label_folder, extension=".txt"):
    label_files = []
    for root, _, files in os.walk(label_folder):
        for file in files:
            if file.endswith(extension):
                label_files.append(os.path.join(root, file))
    return label_files

# Get a list of all label files in the label folder
label_files = get_label_files_recursively(label_folder)

# Randomly shuffle the image files to introduce randomness in the split
random.shuffle(label_files)

# Calculate the number of images for each split
total_images = len(label_files)
train_size = int(train_split * total_images)
val_size = int(val_split * total_images)

# Split the default-dataset into training, validation, and testing sets
train_labels = label_files[:train_size]
val_labels = label_files[train_size:train_size + val_size]
test_labels = label_files[train_size + val_size:]

# Function to copy image and corresponding label files to their respective splits
def copy_files(source_folder, destination_folder, label_paths):
    for label_path in label_paths:
        # Derive image path from the label path
        image_path = label_path.replace("labels", "raw-data")
        image_path = os.path.splitext(image_path)[0] + ".jpg"

        # Get the filenames from the full paths
        image_file = image_path.split('/')[-1]
        label_file = label_path.split('/')[-1]

        # Ignore the "classes.txt" file if present
        if not label_file == "classes.txt":
            # Copy the image and label files to their respective split folders
            shutil.copy(image_path, os.path.join(os.path.join(destination_folder, 'images'), image_file))
            shutil.copy(label_path, os.path.join(os.path.join(destination_folder, 'labels'), label_file))

# Copy images and corresponding label files to the training split
copy_files(image_folder, train_dest, train_labels)

# Copy images and corresponding label files to the validation split
copy_files(image_folder, val_dest, val_labels)

# Copy images and corresponding label files to the testing split
copy_files(image_folder, test_dest, test_labels)
