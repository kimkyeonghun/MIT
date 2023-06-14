import datetime
import os

def make_date_dir(path):
    """
    :param path
    :return: os.path.join(path, date_dir)
    """
    if not os.path.exists(path):
        os.mkdir(path)
    i = 0
    today = datetime.datetime.now()
    name = today.strftime('%Y%m%d')+'-'+'%02d' % i

    while os.path.exists(os.path.join(path, name)):
        i += 1
        name = today.strftime('%Y%m%d')+'-'+'%02d' % i
        
    os.mkdir(os.path.join(path, name))
    return os.path.join(path, name)  