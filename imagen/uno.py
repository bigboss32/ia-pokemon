import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class PokemonMultiLabelDataset(Dataset):
    def __init__(self, root_dir, all_types, transform=None):
        self.root_dir = root_dir
        self.all_types = all_types
        self.transform = transform
        self.samples = []

        for ptype in all_types:
            type_folder = os.path.join(root_dir, ptype)
            if not os.path.isdir(type_folder):
                continue

            for file in os.listdir(type_folder):
                if file.endswith('_artwork.png'):
                    image_path = os.path.join(type_folder, file)
                    info_path = os.path.join(
                        type_folder, file.replace('_artwork.png', '_info.json'))

                    if os.path.exists(info_path):
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        labels = [0] * len(all_types)
                        for t in info['types']:
                            if t in all_types:
                                labels[all_types.index(t)] = 1
                        self.samples.append((image_path, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, labels = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.FloatTensor(labels)
        return image, labels
