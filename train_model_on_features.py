import wandb
from wandb.keras import WandbCallback

wandb.init(project="robot-or-brain-POC", entity="robot-or-brain")

config = {
    "learning_rate": 0.001,
    "epochs": 200,
    "batch_size": 32,
    "validation_path": '../robot_or_brain_combined_data/images_by_class/validation',
    "train_path": '../robot_or_brain_combined_data/images_by_class/train',
}

wandb.config = config


from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory


# ----
# Let's load the data
# ----

train_ds = image_dataset_from_directory(
    config['train_path'],
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
    config['validation_path'],
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
    model.compile(optimizer=Adam(lr=config['learning_rate']), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model(n_classes=len(train_ds.class_names))

# ----
# Define validation metrics
# ----

class PRMetrics(Callback):
  """ Custom callback to compute metrics at the end of each training epoch"""
  def __init__(self, generator=None, num_log_batches=1):
    self.generator = generator
    self.num_batches = num_log_batches
    # store full names of classes
    self.flat_class_names = [k for k, v in generator.class_indices.items()]

  def on_epoch_end(self, epoch, logs={}):
    # collect validation data and ground truth labels from generator
    val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
    val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

    # use the trained model to generate predictions for the given number
    # of validation data batches (num_batches)
    val_predictions = self.model.predict(val_data)
    ground_truth_class_ids = val_labels.argmax(axis=1)
    # take the argmax for each set of prediction scores
    # to return the class id of the highest confidence prediction
    top_pred_ids = val_predictions.argmax(axis=1)

    # Log confusion matrix
    # the key "conf_mat" is the id of the plot--do not change
    # this if you want subsequent runs to show up on the same plot
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            preds=top_pred_ids, y_true=ground_truth_class_ids,
                            class_names=self.flat_class_names)})


# ----
# Let's train the model now
# ----
wandb_callback = WandbCallback(save_weights_only=False, generator=validation_ds, input_type='image', output_type='label',
                         log_evaluation=True, log_evaluation_frequency=5, )
callbacks = [wandb_callback, PRMetrics(validation_ds, config['batch_size'])]

model.fit(
    train_ds,
    epochs=config['epochs'],
    validation_data=validation_ds,
    callbacks=[wandb_callback],
)

model.save('fine_tuned_model_' + wandb.run.id)
