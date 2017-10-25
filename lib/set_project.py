import os
def get_path():
    return os.path.dirname(os.path.abspath(__file__)).rsplit("/",1)[0]
