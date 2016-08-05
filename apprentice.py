from get_data import getData
from model import lstmModel


path = 'images/'
artists = ['pablo_picasso']
url = 'https://en.wikipedia.org/wiki/'

data = getData(path, artists)
data.wiki_scrape(url)
remove = {'pablo_picasso': range(0, 9) + range(14, 41)}
# remove = {'pablo_picasso': [0, 3, 8, 16, 19, 22, 27, 28, 29,
                            # 30] + range(34, 41)}
        #   'claude_monet': [28, 37] + range(47, 69, 2)}
data.remove_files(remove)
pixel_data = data.make_data((128, 128), 16)
# 256 quantized colors, Total colors: 370588
# 128 quantized colors, Total colors: 358275
# 64 quantized colors, Total colors: 341692
# 32 quantized colors, Total colors: 316073
# 16 quantized colors, Total colors: 281398
# 8 quantized colors, Total colors: 237908

model = lstmModel((128, 128), 3, pixel_data)
X, y = model.vectorize()
# model.model(X, y, [0.2, 0.5, 1.0, 1.2])

# data.shape = (81920,)
# X.shape = (27264, 128, 31870)
# y.shape = (27264, 31870)
