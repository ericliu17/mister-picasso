from __future__ import print_function
import numpy as np
import urllib
from bs4 import BeautifulSoup
import os
import shutil
import re
from PIL import Image


class getData(object):

    def __init__(self, path, artists):
        self.path = path
        self.artists = sorted(artists)


    def wiki_scrape(self, url):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        for artist in self.artists:
            webpage = urllib.urlopen(url + artist).read()
            soup = BeautifulSoup(webpage, 'html.parser')
            tags = ['thumbinner', 'gallerybox']
            classes = soup.find_all(True, {'class':tags})
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

        # print 'Renamed {} files in {}.'.format(count, \
        # self.path)


    def make_data(self, size, n_colors):
        filenames = os.listdir(self.path)
        thumb_path = self.path + 'thumbnails/'
        os.makedirs(thumb_path)
        res = size[0] * size[1]
        n_samples = len(filenames)
        data = []

        print('Creating thumbnails...')
        for filename in filenames:
            infile = thumb_path + filename
            self.resize_quantize(filename, size, n_colors)
            img_data = np.asarray(Image.open(infile))
            img_data = img_data.reshape(res, 3)
            img_data = map(tuple, img_data)
            data.extend(img_data)

        return data


    def resize_quantize(self, filename, size, n_colors):
        thumb_path = self.path + 'thumbnails/'
        infile = self.path + filename
        outfile = thumb_path + filename
        img_data = Image.open(infile)
        img_data = img_data.resize(size)
        img_data = img_data.convert('P', palette=Image.ADAPTIVE,
                                    colors=n_colors)
        img_data.convert('RGB').save(outfile)
