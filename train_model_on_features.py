import argparse
from pathlib import Path

from wandb.keras import WandbCallback

import wandb

wandb.init(project="robot-or-brain-POC", entity="robot-or-brain")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data_base_path', type=Path, help='Path to dir containing the metadata csv file.')
args = parser.parse_args()

validation_path = args.data_base_path / 'images_by_class/validation'
train_path = args.data_base_path / 'images_by_class/train'
config = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
    "validation_path": validation_path,
    "train_path": train_path,
    "lr_decay": 1e-4,
    "use_augmentation": True,
}

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

train_datagen = ImageDataGenerator(
    # rescale=1. / 255,
    shear_range=0,
    zoom_range=0,
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1)

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
    fully_connected_output = Dense(1024, activation='relu')(features)
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
    # train_ds,  # .map(lambda x, y: (augment(x), y),),  #Doesn't seem to work correctly yet
    train_datagen.flow_from_directory(
        config['train_path'],
        batch_size=config['batch_size'],
        target_size=(input_resolution, input_resolution),
        class_mode='sparse',
    ),
    epochs=config['epochs'],
    validation_data=validation_ds,
    callbacks=[WandbCallback()],
)

model.save('fine_tuned_model_' + wandb.run.id)
