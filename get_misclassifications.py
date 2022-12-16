import argparse
import pickle
from pathlib import Path

import numpy as np


def get_dataset_with_predictions(dataset_path: Path, model_path: Path):
    """
    Loads a pickled dataset and a model and add the predictions of the model on the dataset clip features to the dataset.
    :param dataset_path:
    :param model_path:
    :return: the initial dataset with extra column 'prediction'
    """
    dataset = _unpickle(dataset_path)
    _add_predictions(dataset, model_path)
    return dataset


def _add_predictions(dataset, model_path):
    model = _unpickle(model_path)
    features = np.stack([v[0] for v in dataset.clip_features])
    dataset['prediction'] = model.predict(features)


def _unpickle(path):
    with open(path, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get list of misclassifications given a model and a pickled dataset.')
    parser.add_argument('--data_set_path', type=Path, help='Path to pickled dataset (e.g. train.pk.)', required=True)
    parser.add_argument('--model_path', type=Path,
                        help='Path to random forest model (e.g. random_forest_model_100.pk).',
                        required=True)
    args = parser.parse_args()

    dataset = get_dataset_with_predictions(args.data_set_path, args.model_path)
    print(dataset)
