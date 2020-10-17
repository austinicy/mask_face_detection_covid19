import os
import datetime
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file):
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)

        image_time = str(datetime.now()).replace("-", "").replace(":", "").replace(" ", "")

        file.save(os.path.join(UPLOAD_FOLDER, image_time + filename))
        return 1
    return 0