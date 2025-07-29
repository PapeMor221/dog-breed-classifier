# src/make_dataset.py

import os
import random
from collections import Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tqdm import tqdm


def load_data(path: str, target_size=(128, 128)) -> tuple:
    images = []
    labels = []
    class_names = []
    labels_ = {}
    current_label = 0

    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            if class_name not in labels_:
                labels_[class_name] = current_label
                class_names.append(class_name)
                current_label += 1
            for img_name in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    try:
                        # Redimensionnement uniforme ici
                        img = cv2.resize(img, target_size)
                        images.append(img)
                        labels.append(labels_[class_name])
                    except Exception as e:
                        print(f"[SKIPPED] {img_path} — erreur resize : {e}")
                        continue

    return np.array(images, dtype="float32"), np.array(labels), class_names



def visualize_class_distribution(labels: np.ndarray, class_names: list):
    class_counts = Counter(labels)
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), tick_label=class_names)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Distribution des classes')
    plt.show()


def plot_images_from_subfolders(base_dir, num_images=3):
    subfolders = [
        os.path.join(base_dir, folder)
        for folder in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, folder))
    ]
    for folder_path in subfolders:
        relative_path = os.path.relpath(folder_path, base_dir)
        print(f"Images de: {relative_path}")
        _, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
        files = os.listdir(folder_path)
        for i in range(num_images):
            img_path = os.path.join(folder_path, files[i])
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].axis('off')
        plt.show()


def img_dimensions(base_dir):
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            print(f'Analyse des images en: {dir_path}')
            files = os.listdir(dir_path)
            dim = []
            for file_name in files:
                img_path = os.path.join(dir_path, file_name)
                img = cv2.imread(img_path)
                if img is not None:
                    height, width, _ = img.shape
                    dim.append((height, width))
            count_dim = Counter(dim)
            print("Dimensions les plus courantes:")
            for dimension, freq in count_dim.most_common(15):
                print(f"Dimension: {dimension}, Fréquence: {freq}")
            print('\n')


def process_dataset(source_dir, processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    data = []
    labels = []

    random.seed(42)

    for class_name in sorted(os.listdir(source_dir)):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        def process_and_save_files(file_list, dest_dir):
            class_dest_dir = os.path.join(dest_dir, class_name)
            os.makedirs(class_dest_dir, exist_ok=True)
            for file in file_list:
                try:
                    image_path = os.path.join(class_dir, file)
                    image = cv2.imread(image_path)
                    image_resized = cv2.resize(image, (128, 128))
                    image_array = img_to_array(image_resized)
                    data.append(image_array)
                    labels.append(class_name)
                    save_path = os.path.join(class_dest_dir, file)
                    cv2.imwrite(save_path, image_resized)
                except Exception as e:
                    print(f"Erreur avec {file}: {e}")
                    continue

        process_and_save_files(files, processed_dir)
        print(f"Classe {class_name}: {len(files)} traitées")

    return np.array(data, dtype="float32"), np.array(labels)


def increase_dataset(data, labels, zoom_range=0.2, horizontal_flip=True, augmentation_ratio=1/3):
    datagen = ImageDataGenerator(
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip
    )

    augmented_images = []
    augmented_labels = []

    augmentation_count = int(len(data) * augmentation_ratio)

    for img, label in zip(data, labels):
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0].astype('uint8'))
            augmented_labels.append(label)
            i += 1
            if i >= (augmentation_count / len(data)):
                break

    final_imgs_data = np.concatenate((data, augmented_images), axis=0)
    final_labels_data = np.concatenate((labels, augmented_labels), axis=0)

    return final_imgs_data, final_labels_data


def split_data(data, labels, val_size=0.2, test_size=0.2, random_state=42):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        random_state=random_state
    )
    val_size_adj = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_size_adj,
        random_state=random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def check_class_distribution(y_data, classes):
    return {cls: np.sum(np.argmax(y_data, axis=1) == cls) for cls in classes}