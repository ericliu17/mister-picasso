from get_data_grayscale import GetData
from rowwise_baseline import LSTMModel
# from rowwise_test import LSTMModel
# from get_data_rgb import GetData
# from pixelwise_baseline import LSTMModel
# from pixelwise_test import LSTMModel

path = 'images/'
artists = ['pablo_picasso']
url = 'https://en.wikipedia.org/wiki/'

data = GetData(path, artists)
# data.scraper(url)
remove = {'pablo_picasso': range(0, 33) + range(34, 41)}
# remove = {'pablo_picasso': [0, 3, 8, 16, 19, 22, 27, 28, 29,
                            # 30] + range(34, 41)}
# data.remove_files(remove)
img_data = data.make_data(size=(128, 128), n_colors=64)


model = LSTMModel(step=3, data=img_data)
# X, y = model.vectorize()
diversities = [1.0]
model.lstm_model(diversities, iterations=100)
