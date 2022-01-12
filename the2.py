import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#BEST CONFIGURATION AS CONFIG
KMEANS = 128
KNN = 32
NOCTAVELAYERS = 12

def get_point_distance(arr1, arr2):
    return np.linalg.norm(arr1-arr2)

def get_clusters(n_clusters, img_list):
    kmeans = KMeans(n_clusters=n_clusters).fit(img_list)
    return kmeans

def get_sift_descriptors(sift, img):
    _, detectors = sift.detectAndCompute(img, None)
    return detectors

def predict_by_knn(k, bow, arr):
    types = bow.keys()
    nns = [99999] * k
    labels = [''] * k
    for t in types:
        for point in bow[t]:
            dist = get_point_distance(arr, point)
            max_n_index = nns.index(max(nns))
            if dist < nns[max_n_index]:
                nns[max_n_index] = dist
                labels[max_n_index] = t

    return nns, labels

def get_acc(bow, kmeans, kk, sift, types, knn):
    count = 0
    correct = 0
    pred_list = []
    truth_list = []
    for t in types:
        fnames = next(os.walk(f'the2_data/validation/{t}'), (None, None, []))[2]
        for file in fnames:
            img = cv2.imread(f'the2_data/validation/{t}/{file}')
            detectors = get_sift_descriptors(sift, img)
            if detectors is not None:
                pred = kmeans.predict(detectors.astype(float))
                h_hist = np.histogram(pred, bins=list(range(kk)))[0]
                a = predict_by_knn(knn, bow, h_hist)[1]
                result = max(set(a), key = a.count)
                count += 1
                if result == t:
                    correct += 1
                pred_list.append(result)
                truth_list.append(t)
    return (correct/count)*100, pred_list, truth_list

def get_bow(types, sift, kmeans, kk):
    bow = {}
    for t in types:
        fnames = next(os.walk(f'the2_data/train/{t}'), (None, None, []))[2]
        hists = []
        for file in fnames:
            img = cv2.imread(f'the2_data/train/{t}/{file}')
            detectors = get_sift_descriptors(sift, img)
            if detectors is not None:
                det_list = detectors.astype(float)
                predictions = kmeans.predict(det_list)
                hist = np.histogram(predictions, bins=list(range(kk)))[0]
                hists.append(hist/np.linalg.norm(hist))
        bow[t] = hists
    return bow

if __name__ == '__main__':
    sift = cv2.SIFT_create(nOctaveLayers=NOCTAVELAYERS)

    types = os.listdir("the2_data/train")
    
    print("Constructing Dictionary...")
    vocab = []
    for t in types:
        fnames = next(os.walk(f'the2_data/train/{t}'), (None, None, []))[2]
        for file in fnames:
            img = cv2.imread(f'the2_data/train/{t}/{file}')
            detectors = get_sift_descriptors(sift, img)
            if detectors is not None:
                for d in detectors:
                    vocab.append(d)
    print("Calculating cluster centers...")
    kmeans = get_clusters(KMEANS, vocab)
    print("Calculating accuracy...")
    acc, pred, truth = get_acc(get_bow(types, sift, kmeans, KMEANS), kmeans, KMEANS, sift, types, KNN)

    print(f"Accuracy is {acc}")
    print(f"Confusion matrix --> \n {confusion_matrix(truth, pred, labels=types)}")