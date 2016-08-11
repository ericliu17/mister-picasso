from __future__ import print_function
import numpy as np
import urllib
from bs4 import BeautifulSoup
import os
import shutil
import re
from PIL import Image
from color_quantization import quantize


class GetData(object):

    def __init__(self, path, artists):
        self.path = path
        self.artists = sorted(artists)


    def scraper(self, url):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        for artist in self.artists:
            webpage = urllib.urlopen(url + artist).read()
            soup = BeautifulSoup(webpage, 'html.parser')
            tags = ['thumbinner', 'gallerybox']
            classes = soup.find_all(True, {'class': tags})
            links = [img['src'] for item in classes
                     for img in item.find_all('img')]
            count = 0

            for link in links:
                path = self.path + artist + '_' + str(count) + '.jpg'
                urllib.urlretrieve('https:' + link, path)
                count += 1

            print('Saved {} {} images in {}.'.format(count, artist,
                                                     self.path))


    def remove_files(self, lst):
        for artist, nums in lst.iteritems():
            for num in nums:
                path = self.path + artist + '_' + str(num) + '.jpg'
                os.remove(path)

            print('Removed {} {} images in {}.'.format(len(nums), artist,
                                                       self.path))

        self.rename_files()


    def make_data(self, size, n_colors):
        filenames = os.listdir(self.path)
        img_data = []
        coll_data = []

        print('Creating {} thumbnails...'.format(len(filenames)))
        for filename in filenames:
            infile = self.path + filename
            img = Image.open(infile)
            img = img.resize(size)
            img = np.asarray(img)
            # Quantize each image
            # img = quantize(img, n_colors)
            img_data.append(img)

        # Quantize all images
        img_data = quantize(img_data, n_colors)
        return np.array(img_data)


    def rename_files(self):
        filenames = os.listdir(self.path)
        temp_path = self.path[:-1] + '_/'
        os.makedirs(temp_path)
        count = 0

        for filename in filenames:
            new_name = re.sub(r'\d+', str(count), filename)
            os.rename(self.path + filename, temp_path + new_name)
            count += 1

        shutil.rmtree(self.path)
        os.rename(temp_path, self.path)
