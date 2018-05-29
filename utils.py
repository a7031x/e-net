import os
import shutil
import ujson as json


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def rmdir(directory):
    try:
        shutil.rmtree(directory)
    except:
        pass
        

def save_json(filename, obj):
    with open(filename, "w") as file:
        json.dump(obj, file)


def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)