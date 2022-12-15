# Robot or brain?

Welcome to the repository for the Robot or Brain project. The purpose of the project was to create a computer vision
model that recognizes the framing behind images depicting AI. More about the project can be read
[here](https://www.esciencecenter.nl/projects/the-robot-or-the-brain-building-a-classifier-for-visual-news-frames-of-artificial-intelligence/)
.

## How to install

Clone this repo:

```shell
git clone git@github.com:robot-or-brain/robot_or_brain.git
```

Install requirements:

```shell
pip install -r requirements.txt
```

## How to run

This project code can be used to train models through a small number of
different ways. In any case, images used for training and validation need to be organized in the expected file
structure, see [organize-images](#organize-images). When that's done, either a resnet-50 model can be fine-tuned
[directly from the images](#Train-resnet50-on-images-(inefficient)) or features from either
the [resnet50](#Fine-tune-ResNet50-using-precomputed-features)
or [CLIP models](#Fine-tune-CLIP-model-using-precomputed-features) can be precomputed once in order to train a
small neural network on top of those features. Precomputing features once is much more energy and time efficient and
also yielded the best results in our project. The main reason to train directly on images however, is that it offers the
possibility to include simple augmentations (scale, shift, flip, etc.) on the images during training. Using these
augmentations had never worked during our project, but would in theory be a technique against overfitting. Finally, we
can create a [zero-shot classifier](#Clip-zero-shot-classifier), without using any train data using the clip model.

### Organize images
For easy loading with Keras and other frameworks, we want images organized in folders as follows.
```
/ root
  /images_by_class
    /train    
      / class_robot
        / file1.jpg
        / file2.jpg
      / class_brain
        / file1.jpg
        / file2.jpg
    /validation
      ..
    /test
      ..
```
However, our dataset was organized as follows:
```
/ root
  / metadata.csv
  / database_a
    / file1.jpg
    / file2.jpg
  / database_b
    / file1.jpg
    / file2.jpg
```
The metadata csv-file contained the following columns:
```"id","imageid","database_name","aiframe","status","coder","coded_time"```
Use `organize_image_folders.py` to get such a dataset reorganized to the keras-ready form as described above.

```shell
python organize_image_folders.py example_data/metadata.csv
```

This will both restructure the data and split data into train, validation and test sets.

## Model creation and validation

A model can be trained on precomputed features (recommended) or directly from the images using a feature extractor model on the fly (inefficient) or using the zero-shot method (bad performance). 

### Precomputing image features

Features can be computed and stored to disk in order to train a model on those afterwards. Images need to be in a directory structure as indicated in [Organize images](#organize-images).

Either a CLIP or ResNet50 model can be used to compute features. Both can be computed one after another, but not during the same run. In this case, both features will be written to their own column in the datafile that is output. Both feature types cannot be processed in a single command because CLIP is using PyTorch and ResNet50 is using Tensorflow, and importing and using both frameworks is problematic.

Note that precomputing features costs a lot of computation, in the order of a 2-3 seconds per image. This procedure was never optimized given the scope and time budget of the project, as ideally it only has to be run once per image ever. This process could be sped up a little by using a GPU. To fully leverage the GPU however, the script should be adapted to load and process batches of images together in parallel. 

```shell
python save_clip_and_resnet_features.py --data_base_path my_data_folder/ --model clip
```

Images will be read from the directory structure under the given folder. Features will be stored in a pickled dataframes in files `train.pk`, `validation.pk` and `test.pk` in the given folder. 


### Train model using precomputed CLIP features

A random forest classifier can be trained on precomputed CLIP features. To do so run:

```shell
python train_random_forest_on_precomputed_features.py --data_base_path=my_data_folder/ --number_of_trees=50
```

or append `> output.txt` if the output needs to be stored in a text file instead of printed to screen.

This will train a classifier, score it using a validation set, print the scores to screen (or to a text file) and save the model in the data folder. 

### train model using precomputed ResNet50 features

A neural network classifier can be trained on the precomputed ResNet50 features (or CLIP features, or both types combined). To do this, run:

```shell
python train_on_precomputed_features.py --data_base_path=data_folder/ --batch_size=32 --feature_type=clip --learning_rate=0.0001 --dropout_rate=0 --epochs=500 --lr_decay=0
```

This will train a model and save it to disk. Note the hyperparameters `learning_rate`, `dropout_rate`, `epochs` and `lr_decay`. Their explanations are outside the scope of this text but can be found in deep learning tutorials.
The model will be written into a folder containing several files. The model can be validated using the notebook [`performance_fine_tune_clip.ipynb`](performance_fine_tune_clip.ipynb).

### Clip zero shot classifier

Using CLIP from openAI, we can do zero-shot classification, meaning that we don't use any training data. This classifier has the worst classification performance by far, but it is amazing that it sort of works at all given the absence of any training on our dataset.
See notebook [`performance_clip_zero_shot.ipynb`](performance_clip_zero_shot.ipynb).

### Train resnet50 on images (inefficient)

A model can be trained directly on the images using a ResNet50 model as a basis. This gives options for fine tuning of the ResNet50 model or using on-the-fly augmentation of images. Neither fine tuning or image augmentation was performed/implemented within this project, so running this doesn't have any benefits over the other options. For this option, using a GPU is highly recommended.

```shell
train_model_on_features.py --batch_size=16 --data_base_path=data_folder/ --dropout_rate=0.95 --epochs=50 --learning_rate=0.0003 --lr_decay=0.0001 --use_augmentation=False
```
Several model files will be saved to a into a folder. The model can be validated using the notebook [`performance_fine_tuned_resnet.ipynb`](performance_fine_tuned_resnet.ipynb).
