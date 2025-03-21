import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.decomposition import PCA

def plot_variance_explained(faces: List[np.ndarray], whiten: bool) -> None:
    """
    Plot variance - number of PCs.
    """
    face_data = np.array([face.flatten() for face in faces])
    max_components = min(face_data.shape[0], face_data.shape[1])
    pca = PCA(n_components=max_components, whiten=whiten)
    pca.fit(face_data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance vs. Number of Principal Components")
    plt.grid(True)
    plt.show()

def compute_pca(faces: List[np.ndarray], num_components: int, whiten: bool) -> Tuple[np.array, List[np.array], List[np.array]]:
    """
    Performs PCA give the number of principle components.
    """
    face_data = np.array([face.flatten() for face in faces])
    pca = PCA(n_components=num_components, whiten=whiten)
    projections = pca.fit_transform(face_data) # faces projected to the eigenspace
    mean_face = pca.mean_
    eigenvectors= pca.components_
    return mean_face, eigenvectors, projections

def vectors_to_images(vectors: List[np.array], std_shape: tuple) -> List[np.array]:
    """
    Change vectors to images for visualization.
    """
    images = []
    for vector in vectors:
        image = vector.reshape(std_shape[0], std_shape[1], std_shape[2])
        image_shifted = image - np.min(image)
        image_scaled = 255 * (image_shifted / np.max(image_shifted))
        image_display = np.round(image_scaled).astype(np.uint8)
        images.append(image_display)
    return images

def faces_reconstruct(projections: np.array, eigenvectors: List[np.array], mean_face: np.array) -> List[np.array]:
    reconstructed = np.dot(projections, eigenvectors) + mean_face
    return reconstructed