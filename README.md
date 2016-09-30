# Mister Picasso

## Summary
Mister Picasso is a deep convolutional neural network that captures pixel and spatial level data to try and paint your portrait in the style of Picasso and friends. It is based on the neural style transfer example from Keras.

Check it out at: http://misterpicasso.net/

## Repo Structure
* app - files for hosting the webapp
    * samples of generated images can be found in `app/static/img/combo/`
* keras - code used from Keras' neural style transfer example, reorganized and a bit optimized
* model
    * `mister_picasso.py` - script used to run model
    * `spatial.py` - code for spatial capture using image patches
* As stated by Keras, you will need to download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in `mister_picasso.py` matches the location of the file in order to run the model.
