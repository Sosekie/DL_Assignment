import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from parsing.arg_parser import ArgParser
from parsing.config_parser import ConfigParser

from utils import get_device, make_reproducible
from data.vocabulary import Vocabulary
from models.utils import get_model_class
from training.checkpointer import ModelCheckpointer


def load_images(image_folder, image_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    images = []
    image_filenames = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            images.append(image)
            image_filenames.append(filename)
    images = torch.stack(images)
    return images, image_filenames


def generate_captions(model, device, images, vocabulary, num_captions=5):
    model.eval()
    generated_captions = []
    with torch.no_grad():
        for i in range(min(num_captions, images.shape[0])):
            generated_caption_tokens = model.generate_image_caption_tokens(image=images[i].unsqueeze(dim=0).to(device))
            print(f"Generated tokens for image {i + 1}: {generated_caption_tokens}")  # Debug information
            generated_caption = " ".join(generated_caption_tokens)
            generated_captions.append(generated_caption)
    return generated_captions


def save_captions(image_filenames, captions, output_file):
    with open(output_file, 'w') as f:
        f.write("image,caption\n")
        for filename, caption in zip(image_filenames, captions):
            f.write(f"{filename},{caption}\n")


def main(args, config):
    make_reproducible(seed=args.seed)

    device = get_device(device_id=args.device_id)

    vocabulary = Vocabulary(captions_file_path=config['vocabulary']['captions_file_path'])

    model = get_model_class(model_name=config['model']['name'])(vocabulary=vocabulary, **config['model']['parameters'])
    model.to(device)
    model.freeze()

    checkpointer = ModelCheckpointer(checkpoint_dir=args.checkpoint_dir, model_config=config['model'])

    last_epoch, model_state_dict = checkpointer.load_checkpoint(device=device)[:2]
    model.load_state_dict(model_state_dict)

    print("Model loaded successfully.")
    print(f"Last epoch: {last_epoch}")

    image_folder = 'flickr8k/test_examples'
    images, image_filenames = load_images(image_folder)

    generated_captions = generate_captions(model, device, images, vocabulary)
    
    for idx, (caption, filename) in enumerate(zip(generated_captions, image_filenames)):
        print(f"Generated caption for {filename}: {caption}")

    output_dir = 'report_utils/generate_caption'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.experiment_name}_generate_captions.txt")
    save_captions(image_filenames, generated_captions, output_file)

    print(f"Captions saved to {output_file}")


if __name__ == "__main__":
    arg_parser = ArgParser()
    args = arg_parser.parse_args()

    config_parser = ConfigParser(config_file_path=args.config_file_path)
    config = config_parser.parse_config_file()

    main(args=args, config=config)
