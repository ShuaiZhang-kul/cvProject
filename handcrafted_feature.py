from matplotlib import pyplot as plt
from skimage.feature import hog
from typing import List
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

def extract_hog_features(face_images: List[np.ndarray], orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm='L2-Hys'):
    """"
    Calculate features using Histogram of Oriented Gradients and its visualisation
    """
    hog_features = []
    hog_visual =[]
    for face_img in face_images:
        if len(face_img.shape) == 3:
            gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = face_img 
        feat, hog_img = hog(gray_img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm=block_norm, visualize = True)
        hog_features.append(feat)
        hog_visual.append(hog_img)
    hog_features = np.array(hog_features)
    return hog_features, hog_visual

def extract_sift_descriptors(face_images: List[np.ndarray]):
    """"
    Calculate features using Scale Invariant Feature Transform
    """
    sift = cv2.SIFT_create() 
    keypoints_list = []
    descriptors_list = []
    for face_img in face_images:
        if len(face_img.shape) == 3:
            gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = face_img 
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)
        if descriptors is None:
            descriptors = np.zeros((0, 128), dtype=np.float32)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    return keypoints_list, descriptors_list

def show_sift_keypoints(face_images: List[np.ndarray], keypoints_list: List[List[cv2.KeyPoint]]):
    """
    Draw images with designated keypoints
    """
    images_per_row=5
    num_images = len(face_images)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    plt.figure(figsize=(15, 3 * num_rows))
    for i, (img, kps) in enumerate(zip(face_images, keypoints_list)):
        plt.subplot(num_rows, images_per_row, i + 1)
        img_with_kps = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_with_kps = cv2.cvtColor(img_with_kps, cv2.COLOR_BGR2RGB)
        plt.imshow(img_with_kps)
        plt.axis('off')
        plt.title(f"SIFT Keypoints {i+1}")
    plt.tight_layout()
    plt.show()

def plot_tsne(features: np.ndarray, labels: np.ndarray, title: str):
    """
    Data visualisation using T-sne
    """
    tsne = TSNE(n_components=2, perplexity=15, random_state=123)
    tsne_2d = tsne.fit_transform(features)
    scatter = plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=labels, cmap='tab10') 
    unique_labels = np.unique(labels)
    handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(label)), label=str(label)) for label in unique_labels]
    plt.legend(handles=handles, title="Class")
    plt.title(title)
    plt.show()

def build_codebook(descriptors_list: List[np.ndarray], n_clusters: int):
    """
    Combine descriptors and create codebook [to create t-sne plot for sift]
    """
    all_descriptors = np.vstack([desc for desc in descriptors_list if desc.size > 0])
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=123)
    kmeans.fit(all_descriptors)
    return kmeans

def compute_bow_histograms(descriptors_list: List[np.ndarray], kmeans) -> np.ndarray:
    """
    Assign the descriptors to clusters and create Bag-of-Visual-Words  [to create -sne plot for sift]
    """
    n_clusters = kmeans.n_clusters
    histograms = []
    for desc in descriptors_list:
        if desc.size == 0:
            hist = np.zeros(n_clusters)
        else:
            words = kmeans.predict(desc)
            hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
            hist = hist.astype(float)
            if hist.sum() > 0:
                hist /= hist.sum()
        histograms.append(hist)
    return np.array(histograms)