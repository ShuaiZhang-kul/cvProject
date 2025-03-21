import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import cv2

def read_faces_from_folder(folder: str = 'Dataset/train') -> List[np.array]:
    """
    Load faces from .npy files stored in a folder.
    """
    files = glob.glob(os.path.join(folder, '*.npy'))
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
    faces = [np.load(file) for file in files]
    return faces

def show_faces(faces: List[np.array]):
    images_per_row = 5
    num_images = len(faces)
    num_rows = num_images // images_per_row + 1

    plt.figure(figsize=(15, 3 * num_rows))

    for i, img in enumerate(faces):
        plt.subplot(num_rows, images_per_row, i + 1)
        img = img[..., ::-1]  
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i + 1}")

    plt.tight_layout()
    plt.show()

def normalize_size(faces: List[np.array]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """
    Rescale all faces matrices to the same dimension therefore they can be concatenated.
    """
    min_height = min(face.shape[0] for face in faces)
    min_width = min(face.shape[1] for face in faces)
    std_faces = [cv2.resize(face, (min_width, min_height)) for face in faces]
    return std_faces, (min_height, min_width)