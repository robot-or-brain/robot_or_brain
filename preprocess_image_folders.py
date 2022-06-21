# Script to transform dataset from form 1 to keras-ready form.
# Form 1:
# / metadata.csv
# / flickr
#   / file1.jpg
#   / file2.jpg
# / imago
#   / file1.jpg
#   / file2.jpg
# With metadata containing the following fields:
# id,"imageid","database_name","aiframe","status","coder","coded_time"
#
# The keras-ready form:
# / class_a
#   / file1.jpg
#   / file2.jpg
# / class_b
#   / file1.jpg
#   / file2.jpg

import csv
from pathlib import Path
import shutil
import os

with open('./20-06-2022-aivisuals-10-41manual_coding.csv','r') as metadata:
    files = [i for i in csv.DictReader(metadata)]
extensions = set([f['imageid'].split('.')[-1] for f in files])

ps = [(Path(f['database_name']) / f['imageid'], Path(f['aiframe']), f['id']) for f in files]

for source_path, folder, id in ps:
    if os.path.exists(source_path):
        pass
    else:
        print('missing', source_path)
        continue
    os.makedirs(folder, exist_ok=True)
    target_path = folder / (id + '.jpg')
    shutil.copy(source_path, target_path)
