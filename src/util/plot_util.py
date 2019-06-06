import numpy as np
from pylab import *

class IndexColormap():
    def __init__(self, colormap='viridis', size=20):
        self.size = size
        self.colors = []

        cmap = cm.get_cmap(colormap, size)    # PiYG
        for i in range(cmap.N):
            rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
            self.colors.append((matplotlib.colors.rgb2hex(rgb)))

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        assert 0 <= key < self.size, 'Key out of range'
        return self.colors[key]
