# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

from __future__ import print_function
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def quantize(data, n_colors):
    data = np.array(data, dtype=float)
    # Load Image and transform to a 2D numpy array.
    w, h, d = data.shape
    assert d == 3
    image_array = np.reshape(data, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)

    return recreate_image(kmeans.cluster_centers_, labels, w, h)


def recreate_image(centroids, labels, w, h):
    # Recreate the (compressed) image from the centroids & labels
    d = centroids.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = centroids[labels[label_idx]]
            label_idx += 1
    return image
