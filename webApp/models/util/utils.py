import os
import configparser
from werkzeug.utils import secure_filename

def get_conf(key, value):
    cf=configparser.ConfigParser()
    cf.read('util\\config.ini')
    return cf.get(key, value)

def get_file_path(folder, filename):
    return os.path.join(folder, filename)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file):
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return 1
    return 0

if __name__ == '__main__':
    print('common function')