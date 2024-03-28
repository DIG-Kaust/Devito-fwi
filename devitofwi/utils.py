from os import listdir, path
from shutil import rmtree
from tempfile import gettempdir


def clear_devito_cache():
    tempdir = gettempdir()
    for i in listdir(tempdir):
        if i.startswith('devito-'):
            try:
                target = path.join(tempdir, i)
                rmtree(target)
            except:
                pass
