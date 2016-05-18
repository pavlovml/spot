from .error import SpotError
import os, os.path, fnmatch, json

def load_json_file(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except ValueError:
        raise SpotError("Could not decode JSON file '" + filename + "'")

def glob_keyed_files(path, ext):
    res = {}
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.' + ext):
            key = os.path.splitext(os.path.basename(filename))[0]
            res[key] = os.path.join(root, filename)
    return res

def mkdirp(path):
    if not os.path.exists(path):
        os.makedirs(path)
