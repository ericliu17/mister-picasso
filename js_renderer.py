import os
from selenium import webdriver
from bs4 import BeautifulSoup
from requests import get

path = 'images/'
artists = ['pablo-picasso']
url = 'http://www.wikiart.org/en/'


browser = webdriver.Chrome() # Instantiate a webdriver object
browser.get(url + artists[0])
# Makes list of links to get full image
links = []
# This is the container of images on the main page
cards = browser.find_elements_by_class_name('st-Masonry-container')
for img_src in cards:
    # Assemble list to pass to requests and beautifulsoup
    links.append(img_src.get_attribute('src'))
