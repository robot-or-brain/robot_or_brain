import argparse
from pathlib import Path

import wandb
from wandb.keras import WandbCallback
import numpy as np

wandb.init(project="robot-or-brain-POC", entity="robot-or-brain")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data_base_path', type=Path, help='Path to dir containing the metadata csv file.')
args = parser.parse_args()

validation_path = args.data_base_path / 'images_by_class/validation'
train_path = args.data_base_path / 'images_by_class/train'
config = {
    "learning_rate": 0.001,
    "epochs": 200,
    "batch_size": 32,
    "validation_path": validation_path,
    "train_path": train_path,
    "lr_decay": 1e-4,
}

wandb.config = config

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras import layers

# ----
# Let's load the data
# ----

input_resolution = 224

augment = tf.keras.Sequential([
    layers.Resizing(input_resolution, input_resolution),
    layers.Rescaling(1. / 255),
    layers.RandomFlip(mode='horizontal'),
    # layers.RandomRotation(),
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


def create_model(n_classes):
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling='avg'
    )

    features = base_model.output
    fully_connected_output = Dense(1024, activation='relu')(features)
    predictions = Dense(n_classes, activation='softmax')(fully_connected_output)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # Training only top layers i.e. the layers which we have added in the end
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=config['learning_rate'], decay=config['lr_decay']), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model(n_classes=len(train_ds.class_names))

# ----
# Let's train the model now
# ----
wandb_callback = WandbCallback(save_weights_only=False, generator=validation_ds, input_type='image',
                               output_type='label',
                               log_evaluation=True, log_evaluation_frequency=20, )
callbacks = [wandb_callback]

model.fit(
    train_ds, #.map(lambda x, y: (augment(x), y),),  #Doesn't seem to work corretly yet
    epochs=config['epochs'],
    validation_data=validation_ds,
    callbacks=[wandb_callback],
)

model.save('fine_tuned_model_' + wandb.run.id)
