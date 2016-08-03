import numpy as np
from PIL import Image
import urllib
from bs4 import BeautifulSoup
import os
import shutil


class getData(object):

    def __init__(self, paths, prefix):
        self.paths = paths
        self.thumb_paths = [path + 'thumbnails/' for path in self.paths]
        self.prefixes = [prefix + '_' for prefix in prefixes]
        self.number = len(self.paths)


    def wiki_scrape(self, urls):
        for i in xrange(self.number):
            os.makedirs(self.paths[i])
            data = urllib.urlopen(urls[i]).read()
            soup = BeautifulSoup(data, 'html.parser')
            classes = soup.find_all(True, {'class':['thumbinner',
                                                    'gallerybox']})
            links = [img['src'] for item in classes
                     for img in item.find_all('img')]

            count = 0
            for link in links:
                urllib.urlretrieve('https:' + link,
                                   self.paths[i] + self.prefixes[i] + \
                                   str(count) + '.jpg')
                count += 1

            print 'Saved {} files in {}.'.format(count, self.paths[i])


    def remove_files(self, lst):
        for i in xrange(self.number):
            for num in lst[i]:
                filename = self.paths[i] + self.prefixes[i] + \
                           str(num) + '.jpg'
                os.remove(filename)

            print 'Removed {} files in {}.'.format(len(lst[i]), self.paths[i])

        self.rename_files()


    def rename_files(self):
        for i in xrange(self.number):
            filenames = os.listdir(self.paths[i])
            if filenames[0] == '.DS_Store':
                filenames.pop(0)

            temp_path = self.paths[i][:-1] + '_/'
            os.makedirs(temp_path)

            count = 0
            for filename in filenames:
                os.rename(self.paths[i] + filename,
                          temp_path + self.prefixes[i] + str(count) + '.jpg')
                count += 1

            shutil.rmtree(self.paths[i])
            os.rename(temp_path, self.paths[i])

            # print 'Renamed {} files in {}.'.format(count, \
            # self.paths[i])


    def resize_imgs(self, size):
        for i in xrange(self.number):
            filenames = os.listdir(self.paths[i])
            os.makedirs(self.thumb_paths[i])

            for filename in filenames:
                infile = self.paths[i] + filename
                outfile = self.thumb_paths[i] + filename[:-4] + '.jpg'
                im = Image.open(infile).convert('RGB')
                im.thumbnail(size, Image.ANTIALIAS)
                im.save(outfile, 'JPEG')

            print 'Created {} thumbnails in {}.'.format(len(filenames),
                                                        self.thumb_paths[i])


    def vect_img(self, print_shape=False):
        all_vectors = []
        for i in xrange(self.number):
            filenames = os.listdir(self.thumb_paths[i])

            vectors = []
            for filename in filenames:
                img = Image.open(self.thumb_paths[i] + filename)
                vectors.append(np.asarray(img))

            all_vectors.append(vectors)

            print 'Vectorized {} files in {}.'.format(len(filenames),
                                                      self.thumb_paths[i])
            if print_shape:
                for j in all_vectors[i]:
                    print j.shape

        return all_vectors


if __name__ == '__main__':
    paths = ['images/picasso/', 'images/monet/']
    prefixes = ['picasso', 'monet']
    urls = ['https://en.wikipedia.org/wiki/Pablo_Picasso',
            'https://en.wikipedia.org/wiki/Claude_Monet']

    data = getData(paths, prefixes)
    data.wiki_scrape(urls)
    remove = [[0, 3, 27, 28, 34, 35, 36, 37, 38, 39, 40],
              xrange(47, 69, 2)]
    data.remove_files(remove)
    data.resize_imgs((128,128))
    a = data.vect_img()
