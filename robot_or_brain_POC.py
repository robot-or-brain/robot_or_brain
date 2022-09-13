import wandb
from wandb.keras import WandbCallback

wandb.init(project="robot-or-brain-POC", entity="robot-or-brain")

config = {
    "learning_rate": 0.001,
    "epochs": 200,
    "batch_size": 32,
}

wandb.config = config

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory

base_model = ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling='avg'
)

features = base_model.output
fully_connected_output = Dense(1024, activation='relu')(features)
predictions = Dense(8, activation='softmax')(fully_connected_output)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Training only top layers i.e. the layers which we have added in the end
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=config['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# ----
# Let's train the model now
# ----

train_ds = image_dataset_from_directory(
    '../robot_or_brain_data/images_by_class/train',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=config['batch_size'],
    image_size=(224, 224),
    shuffle=True,
    seed=0,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

validation_ds = image_dataset_from_directory(
    '../robot_or_brain_data/images_by_class/validation',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=config['batch_size'],
    image_size=(224, 224),
    shuffle=True,
    seed=0,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

model.fit(
    train_ds,
    epochs=config['epochs'],
    validation_data=validation_ds,
    callbacks=[WandbCallback()],
)

model.save('fine_tuned_model')
