import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import load_dataset

parser = argparse.ArgumentParser(description='Train classifier on directory structure with images.')
parser.add_argument('--data_base_path', type=Path, help='Path to dir containing the metadata csv file.', required=True)
parser.add_argument('--model', type=str, help='Choose from "clip" or "resnet".', required=True)

args = parser.parse_args()


def predict_image_with_clip(image_path):
    import torch
    model, preprocess, device = ClipModel().get_model_and_preprocess()

    img = Image.open(image_path)
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image = model.encode_image(image)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return logits_per_image.cpu().numpy()


class ClipModel:

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device

    def get_model_and_preprocess(self):
        if self.model is None:
            self.initialize_model()
        return self.model, self.preprocess, self.device

    def initialize_model(self):
        import torch
        import clip
        # See first example at https://github.com/openai/CLIP#usage
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)


def predict_image_with_resnet(image_path):
    model = ResnetModel().get_model()

    img = Image.open(image_path)
    np_array = np.expand_dims(np.array(img), axis=0)

    return model.predict(np_array)


class ResnetModel:
    def __init__(self):
        self.model = None

    def get_model(self):
        if self.model is None:
            self.initialize_model()
        return self.model

    def initialize_model(self):
        import tensorflow as tf
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        from tensorflow.keras.models import Model
        base_model = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling='avg'
        )
        input_layer = tf.keras.Input(shape=(None, None, 3))
        preprocessed_input = preprocess_input(input_layer)
        prediction_layer = base_model(preprocessed_input)
        self.model = Model(inputs=input_layer, outputs=prediction_layer)


def encode_and_save(base_dir, split):
    data_set_path = base_dir / f'{split}.pk'

    if data_set_path.exists():
        data_set = pd.read_pickle(data_set_path)
    else:
        data_set, _ = load_dataset(split, base_dir=base_dir / 'images_by_class')

    if args.model == 'clip':
        data_set['clip_features'] = [predict_image_with_clip(p) for p in tqdm(data_set['paths'])]

    if args.model == 'resnet':
        data_set['resnet_features'] = [predict_image_with_resnet(p) for p in tqdm(data_set['paths'])]

    data_set.to_pickle(data_set_path)


encode_and_save(args.data_base_path, 'validation')
encode_and_save(args.data_base_path, 'train')
