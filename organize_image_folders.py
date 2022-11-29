description = """ 
Script to transform dataset from form 1 to keras-ready form.
Form 1:
/ metadata.csv
/ flickr
  / file1.jpg
  / file2.jpg
/ imago
  / file1.jpg
  / file2.jpg
With metadata containing the following fields:
"id","imageid","database_name","aiframe","status","coder","coded_time"
The keras-ready form:
/ class_a
  / file1.jpg
  / file2.jpg
/ class_b
  / file1.jpg
  / file2.jpg
"""

import csv
from pathlib import Path
import shutil
import os
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('csv_path', type=Path, help='Path to the csv file containing the metadata for the images.')
args = parser.parse_args()

with open(args.csv_path, 'r') as metadata:
    files = [i for i in csv.DictReader(metadata)]
extensions = set([f['imageid'].split('.')[-1] for f in files])

instances = [(f['database_name'], f['imageid'], f['aiframe'], f['id']) for f in files]
rest, test = train_test_split(instances, test_size=0.25, random_state=0, stratify=[y for db, _, y, _ in instances])
train, validation = train_test_split(rest, test_size=0.2, random_state=0, stratify=[y for db, _, y, _ in rest])


def organize_file_structure(instances, split, base_path):
    print('Creating {} folder with {} instances'.format(split, len(instances)))
    missing = []
    for database_name, source_file, class_name, id in instances:
        source_path = base_path / database_name / source_file
        if not os.path.exists(source_path):
            missing += [source_path]
            continue

        class_folder_name = make_class_name_suitable_for_file_paths(class_name)

        target_folder = Path('images_by_class') / split / class_folder_name
        os.makedirs(target_folder, exist_ok=True)
        target_path = target_folder / (id + '.jpg')
        shutil.copy(source_path, target_path)
    if missing:
        print('The following image files where missing: \n', '\n'.join([str(m) for m in missing]))


def make_class_name_suitable_for_file_paths(class_name):
    class_name_without_slashes = ' or '.join(class_name.split('/'))
    return ' '.join(class_name_without_slashes.split('  '))


base_path = Path(args.csv_path).parent
organize_file_structure(train, 'train', base_path)
organize_file_structure(validation, 'validation', base_path)
organize_file_structure(test, 'test', base_path)
