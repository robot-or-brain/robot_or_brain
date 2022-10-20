import argparse
from pathlib import Path

from tqdm import tqdm

from utils import load_dataset

parser = argparse.ArgumentParser(description='Train classifier on directory structure with images.')
parser.add_argument('--data_base_path', type=Path, help='Path to dir containing the metadata csv file.', required=True)

args = parser.parse_args()

from PIL import Image
import clip
import torch


def predict_image_with_clip(image_path):
    # See first example at https://github.com/openai/CLIP#usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    img = Image.open(image_path)
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image = model.encode_image(image)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs


def encode_and_save(base_dir, split):
    data_set, _ = load_dataset(split, base_dir=base_dir / 'images_by_class')
    encodings = [predict_image_with_clip(p) for p in tqdm(data_set['paths'])]
    data_set['clip_features'] = encodings
    data_set.to_pickle(base_dir / f'{split}.pk')


encode_and_save(args.data_base_path, 'validation')
encode_and_save(args.data_base_path, 'train')
