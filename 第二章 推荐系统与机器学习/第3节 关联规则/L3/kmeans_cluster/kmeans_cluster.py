import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt

def load_data(filePath:str):
    f = open(filePath, 'rb')
    img = np.array( Image.open(f) )
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(img.reshape(img.shape[0]*img.shape[1], img.shape[2]))
    return data, img.shape[:2]

def cluster(data, size):
    kmeans = KMeans(n_clusters=2)
    print("Start segmentation")
    kmeans.fit(data)
    label = kmeans.predict(data)
    print("Segmentation result")
    label = label.reshape(size)
    plt.imshow(label)
    plt.show()

if __name__ == '__main__':
    filepath = '3.jpg'
    data, size = load_data(filepath)
    cluster(data, size)