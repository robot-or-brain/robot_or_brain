import argparse
from pathlib import Path

from wandb.keras import WandbCallback

import wandb

wandb.init(project="robot-or-brain-POC", entity="robot-or-brain")

parser = argparse.ArgumentParser(description='Train classifier on directory structure with images.')
parser.add_argument('data_base_path', type=Path, help='Path to dir containing the metadata csv file.')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=32, type=int, help='Number of images used each time to calculate the gradient.')
parser.add_argument('--learning_rate', default=0.001, type=float, help='The size of the update step during learning.')
parser.add_argument('--lr_decay', default=1e-4, type=float, help='The rate at which the learning rate is decreased over epochs.')
parser.add_argument('--dropout_rate', default=None, type=float, help='Rate for dropout layers. None means no dropout.')
parser.add_argument('--use_augmentation', default=True, choices=('True', 'False'), help='Should images be augmented by random zooming, rotating etc.')


args = parser.parse_args()

validation_path = args.data_base_path / 'images_by_class/validation'
train_path = args.data_base_path / 'images_by_class/train'
config = {
    "learning_rate": args.learning_rate,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "validation_path": validation_path,
    "train_path": train_path,
    "lr_decay": args.lr_decay,
    "use_augmentation": args.use_augmentation,
    "dropout_rate": args.dropout_rate,
}

print(config)
wandb.config = config

from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers

# ----
# Let's load the data
# ----

input_resolution = 244
augment = tf.keras.Sequential([
    layers.RandomContrast(0.1, seed=None),
    layers.RandomBrightness(0.1, value_range=(0, 255)),
    layers.RandomFlip(mode='horizontal'),
    layers.RandomRotation(factor=.02),
    tf.keras.layers.RandomTranslation(
        0.02,
        0.02,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
    ),

    layers.RandomZoom(
        (-0.2, 0),
        width_factor=None,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
    ),
    layers.Resizing(input_resolution, input_resolution),
])

train_ds = image_dataset_from_directory(
    config['train_path'],
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=config['batch_size'],
    image_size=(input_resolution, input_resolution),
    shuffle=True,
    seed=0,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

validation_ds = image_dataset_from_directory(
    config['validation_path'],
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=config['batch_size'],
    image_size=(input_resolution, input_resolution),
    shuffle=True,
    seed=0,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


def create_model(n_classes, use_augmentation):
    """
    Creates a model. Optionally, augmentation layers can be added before the
    rest of the model. Note that these are only run during training
    (when training=True is passed to them). During prediction mode, augmentation
    is always turned off.
    :param n_classes:
    :param use_augmentation: Whether to include augmentation layers in the model
    :return:
    """
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling='avg'
    )

    input_layer = tf.keras.Input(shape=(None, None, 3))
    if use_augmentation:
        inputs = augment(input_layer)
    else:
        inputs = input_layer
    preprocessed_input = preprocess_input(inputs)
    features = base_model(preprocessed_input)

    fully_connected_layer = Dense(1024, activation='relu')(features)
    if config['dropout_rate'] is not None:
        dropout = tf.keras.layers.Dropout(config['dropout_rate'])
        fully_connected_output = dropout(fully_connected_layer)
    else:
        fully_connected_output = fully_connected_layer

    predictions = Dense(n_classes, activation='softmax')(fully_connected_output)
    # this is the model we will train
    model = Model(inputs=input_layer, outputs=predictions)
    # Training only top layers i.e. the layers which we have added in the end
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=config['learning_rate'], decay=config['lr_decay']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model(n_classes=len(train_ds.class_names), use_augmentation=config['use_augmentation'])

# ----
# Let's train the model now
# ----

model.fit(
    train_ds,
    epochs=config['epochs'],
    validation_data=validation_ds,
    callbacks=[WandbCallback()],
)

model.save('fine_tuned_model_' + wandb.run.id)
