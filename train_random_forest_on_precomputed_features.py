import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from utils import print_performance_metrics

parser = argparse.ArgumentParser(description='Train classifier on directory structure with images.')
parser.add_argument('--data_base_path', type=Path, help='Path to dir containing the metadata csv file.')
parser.add_argument('--number_of_trees', default=1000, type=int, help='Number of random trees to learn.')
parser.add_argument('--feature_type', default='both', choices=['clip', 'resnet', 'both'],
                    help='Use only clip or resnet features or both.')

args = parser.parse_args()

validation_path = args.data_base_path / 'validation.pk'
train_path = args.data_base_path / 'train.pk'


def score(path, model):
    print(f'Scoring {path}')
    vc = pd.read_pickle(path)
    vclipc = np.stack([fv[0] for fv in vc.clip_features.to_numpy()], axis=0)
    rf_pred_clipvc = model.predict(vclipc)
    trues = vc.y
    predictions = rf_pred_clipvc
    class_list = (sorted(set(list(predictions) + list(trues))))

    print(class_list)
    print(confusion_matrix(trues, predictions))
    print_performance_metrics(trues, predictions, class_list)


def train(path, n_trees=500):
    print(f'Training on {path}')
    clf = RandomForestClassifier(n_estimators=n_trees)
    c = pd.read_pickle(path)
    c['names'] = [str(p).split('\\')[-1] for p in c.paths]
    clipc = np.stack([c.loc[c['names'] == n].clip_features.to_numpy()[0][0] for n in c.names])
    yc = np.stack([c.loc[c['names'] == n].y for n in c.names])[:, 0]
    return clf.fit(clipc, yc)


model = train(train_path, args.number_of_trees)
score(train_path, model)
score(validation_path, model)
