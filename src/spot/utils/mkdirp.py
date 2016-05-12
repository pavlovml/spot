import os
import os.path

def mkdirp(path):
    if not os.path.exists(path):
        os.makedirs(path)
