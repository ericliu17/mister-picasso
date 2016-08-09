from get_data import getData
from model import lstmModel


path = 'images/'
artists = ['pablo_picasso']
url = 'https://en.wikipedia.org/wiki/'

data = getData(path, artists)
# data.wiki_scrape(url)
# remove = {'pablo_picasso': range(0, 9) + range(14, 41)}
# remove = {'pablo_picasso': [0, 3, 8, 16, 19, 22, 27, 28, 29,
                            # 30] + range(34, 41)}
        #   'claude_monet': [28, 37] + range(47, 69, 2)}
# data.remove_files(remove)
img_data = data.make_data(size=(128, 128), n_colors=64)

model = lstmModel(step=10, data=img_data)
X, y = model.vectorize()
diversities = [1.0]
model.nn_model(X, y, diversities, iterations=100)
