import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm
from wandb.keras import WandbCallback

import wandb

wandb.init(project="clip-features", entity="robot-or-brain")

parser = argparse.ArgumentParser(description='Train classifier on directory structure with images.')
parser.add_argument('--data_base_path', type=Path, help='Path to dir containing the metadata csv file.')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Number of images used each time to calculate the gradient.')
parser.add_argument('--learning_rate', default=0.0003, type=float, help='The size of the update step during learning.')
parser.add_argument('--lr_decay', default=0.0001, type=float,
                    help='The rate at which the learning rate is decreased over epochs.')
parser.add_argument('--dropout_rate', default=0.85, type=float, help='Rate for dropout layers. None means no dropout.')
parser.add_argument('--feature_type', default='both', choices=['clip', 'resnet', 'both'], help='Use only clip or resnet features or both.')

args = parser.parse_args()

if args.feature_type != 'clip':
    raise NotImplementedError(f"Only clip feature_type is implemented at the moment but the user selected '{args.feature_type}' feature type instead.")

validation_path = args.data_base_path / 'validation.pk'
train_path = args.data_base_path / 'train.pk'

config = {
    "learning_rate": args.learning_rate,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "validation_path": validation_path,
    "train_path": Path(''),
    "lr_decay": args.lr_decay,
    "dropout_rate": args.dropout_rate,
}

print(config)

wandb.config = config


def create_model(n_classes, n_features):
    """
    Creates a model. Optionally, augmentation layers can be added before the
    rest of the model. Note that these are only run during training
    (when training=True is passed to them). During prediction mode, augmentation
    is always turned off.
    :param n_features:
    :param n_classes:
    :return:
    """

    model = Sequential()
    print(f"{n_features=}")
    model.add(Input(shape=n_features))
    model.add(Dense(1024, activation='relu'))
    if config['dropout_rate'] is not None:
        model.add(tf.keras.layers.Dropout(config['dropout_rate']))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=config['learning_rate'], decay=config['lr_decay']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def load_dataset(path):
    data = pd.read_pickle(path)
    n_features = data['encodings'][0].shape[-1]
    class_names = data['y'].unique()
    y = np.array([np.where(class_names == e)[0][0] for e in data['y']])
    x = np.concatenate(data['encodings'].to_numpy())
    print(f'Loaded {len(x)} instances from {path}.')
    print(f'X shape {x.shape} and y shape {y.shape} with labels:\n{data["y"].value_counts()}.')
    train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(config['batch_size'])
    return train_ds, class_names, n_features


train_ds, class_names, n_features = load_dataset(train_path)
val_ds, _, _ = load_dataset(validation_path)

print(train_ds)

model = create_model(n_classes=(len(class_names)), n_features=n_features)
print(model.summary())

model.fit(
    train_ds,
    epochs=config['epochs'],
    validation_data=val_ds,
    callbacks=[WandbCallback(save_model=False)],
)

model.save('clip_features_model_' + wandb.run.id)


def evaluate(model, validation_ds, class_names):
    predicted = [class_names[v] for v in np.argmax(model.predict(validation_ds), 1)]
    trues = [class_names[int(y)] for _x, y in validation_ds.unbatch()]

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    confusion = confusion_matrix(trues, predicted)
    print(confusion)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=class_names)
    # _ = disp.plot(cmap='Greys', xticks_rotation='vertical')

    # Confusion matrix number 2
    unique_label = np.unique([trues, predicted])
    cmtx = pd.DataFrame(
        confusion_matrix(trues, predicted, labels=unique_label),
        index=['true:{:}'.format(x) for x in unique_label],
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    print(cmtx)


evaluate(model, val_ds, class_names)
