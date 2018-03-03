import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

la = [[1,2,3],[4,5,6]]
lb = [1,-1]
a = np.array(la)
b = np.array(lb).reshape(2,1)

print(a*b)