import argparse
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from utils import print_performance_metrics, load_clip_features_and_labels_from_dataset

parser = argparse.ArgumentParser(
    description='Trains a random forest classifier on directory containing precomputed image feature vectors.')
parser.add_argument('--data_base_path', type=Path, help='Path to dir containing validation.pk and train.pk.')
parser.add_argument('--number_of_trees', default=1000, type=int, help='Number of random trees to learn.')

args = parser.parse_args()

validation_path = args.data_base_path / 'validation.pk'
train_path = args.data_base_path / 'train.pk'
model_path = args.data_base_path / f'random_forest_model_{args.number_of_trees}.pk'


def score(path: str, model: RandomForestClassifier):
    """
    Validate the model on the dataset at the given path. Results are printed to screen.
    :param path: Path of the dataset to use for scoring.
    :param model: Model to be validated.
    :return: None
    """
    print(f'Scoring {path}')
    X, y = load_clip_features_and_labels_from_dataset(path)
    predictions = model.predict(X)
    class_list = (sorted(set(list(predictions) + list(y))))

    print(class_list)
    print(confusion_matrix(y, predictions))
    print_performance_metrics(y, predictions, class_list)


def train(path: str, n_trees: int = 500) -> RandomForestClassifier:
    """
    Train a random forest classifier on a features from a stored dataset.
    :param path: path of the pickled dataset containing precomputed features and labels.
    :param n_trees: Hyperparameter for the random forest.
    :return: trained classifier
    """
    print(f'Training on {path}')
    X, y = load_clip_features_and_labels_from_dataset(path)
    clf = RandomForestClassifier(n_estimators=n_trees)
    return clf.fit(X, y)


model = train(train_path, args.number_of_trees)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
score(train_path, model)
score(validation_path, model)
