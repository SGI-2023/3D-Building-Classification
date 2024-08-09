import os.path as osp
import os
from tqdm import tqdm
import random 
import torch 

print(os.getcwd())

import sys
sys.path.append('..')


from torchvision import datasets


from diffusion.utils import DATASET_ROOT, get_classes_templates
from diffusion.dataset.objectnet import ObjectNetBase
from diffusion.dataset.imagenet import ImageNet as ImageNetBase
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms
from torchvision.datasets.folder import default_loader

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_to_idx=None, transform=None):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.imgs = self._load_imgs()
        self.loader = default_loader  # Default loader for image files

    def _load_imgs(self):
      imgs = []
      for root, dirs, files in os.walk(self.root_dir):
          for file in files:
              if file.endswith(('.png', '.jpg', '.jpeg')):  # check if the file is an image
                  path = os.path.join(root, file)
                  unique_id, class_label = self._get_unique_id_from_path(path)
                  # Only add if class_label exists in class_to_idx
                  if self.class_to_idx is None or class_label in self.class_to_idx:
                      imgs.append((path, unique_id, self.class_to_idx[class_label]))
      return imgs

    def _get_unique_id_from_path(self, file_path):
      # Extract the parts of the file path
      path_parts = file_path.split(os.sep)
      
      # Assuming the class name is the second-to-last folder in the path and the model ID is the last folder
      class_name = path_parts[-4]
      model_id = path_parts[-3]
      difficulty = path_parts[-2]  # This will be either 'easy' or 'hard'
      #unique_id = f"{class_name}_{model_id}"
      unique_id = f"{class_name}_{model_id}_{difficulty}"
      
      return unique_id, class_name

    def __getitem__(self, idx):
      image_path, unique_id, class_label = self.imgs[idx]
      _, class_label = self._get_unique_id_from_path(image_path)  # Get class label
      
      # Convert class label to class index if mapping is provided
      class_idx = self.class_to_idx[class_label] if self.class_to_idx else -1
      
      image = self.loader(image_path)
      if self.transform:
          image = self.transform(image)
      
      return image, class_idx, unique_id  # Returns image, class index, and unique_id


    def __len__(self):
        return len(self.imgs)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class MNIST(datasets.MNIST):
    """Simple subclass to override the property"""
    class_to_idx = {str(i): i for i in range(10)}


def get_target_dataset(name: str, train=False, transform=None, target_transform=None):
    """Get the torchvision dataset that we want to use.
    If the dataset doesn't have a class_to_idx attribute, we add it.
    Also add a file-to-class map for evaluation
    """

    if name == "cifar10":
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, transform=transform,
                                   target_transform=target_transform, download=True)
    elif name == "stl10":
        dataset = datasets.STL10(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                 target_transform=target_transform, download=True)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    elif name == "pets":
        dataset = datasets.OxfordIIITPet(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                         target_transform=target_transform, download=True)

        # lower case every key in the class_to_idx
        dataset.class_to_idx = {k.lower(): v for k, v in dataset.class_to_idx.items()}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._images, dataset._labels)}
    elif name == "flowers":
        dataset = datasets.Flowers102(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                      target_transform=target_transform, download=True)
        classes = list(get_classes_templates('flowers')[0].keys())  # in correct order
        dataset.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)}
    elif name == "aircraft":
        dataset = datasets.FGVCAircraft(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                        target_transform=target_transform, download=True)

        # replace backslash with underscore -> need to be dirs
        dataset.class_to_idx = {
            k.replace('/', '_'): v
            for k, v in dataset.class_to_idx.items()
        }

        dataset.file_to_class = {
            fn.split("/")[-1].split(".")[0]: lab
            for fn, lab in zip(dataset._image_files, dataset._labels)
        }
        # dataset.file_to_class = {
        #     fn.split("/")[-1].split(".")[0]: lab
        #     for fn, lab in zip(dataset._image_files, dataset._labels)
        # }

    elif name == "food":
        dataset = datasets.Food101(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                   target_transform=target_transform, download=True)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    elif name == "eurosat":
        if train:
            raise ValueError("EuroSAT does not have a train split.")
        dataset = datasets.EuroSAT(root=DATASET_ROOT, transform=transform, target_transform=target_transform,
                                   download=True)
    elif name == 'imagenet':
        assert not train
        base = ImageNetBase(transform, location=DATASET_ROOT)
        dataset = datasets.ImageFolder(root=osp.join(DATASET_ROOT, 'imagenet/val'), transform=transform,
                                       target_transform=target_transform)
        dataset.class_to_idx = None  # {cls: i for i, cls in enumerate(base.classnames)}
        dataset.classes = base.classnames
        dataset.file_to_class = None
    elif name == 'objectnet':
        base = ObjectNetBase(transform, DATASET_ROOT)
        dataset = base.get_test_dataset()
        dataset.class_to_idx = dataset.label_map
        dataset.file_to_class = None  # todo
    elif name == "caltech101":
        if train:
            raise ValueError("Caltech101 does not have a train split.")
        dataset = datasets.Caltech101(root=DATASET_ROOT, target_type="category", transform=transform,
                                      target_transform=target_transform, download=True)

        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.categories)}
        dataset.file_to_class = {str(idx): dataset.y[idx] for idx in range(len(dataset))}
    elif name == "mnist":
        dataset = MNIST(root=DATASET_ROOT, train=train, transform=transform, target_transform=target_transform,
                        download=True)
                        
    elif name == "buildingnet":
        dataset = datasets.ImageFolder(root= "/content/gdrive/MyDrive/diffusion-classifier-master/buildingnet4",
        transform=transform)
        
    elif name == "shapenet":
      
      class_to_idx = {'airplane': 0, 'car': 1, 'chair': 2} 
      dataset = ShapeNetDataset(root_dir="/content/drive/MyDrive/shapenet_dataset", 
                          class_to_idx=class_to_idx, 
                          transform=transform)
      
      # This will ensure that only valid data entries are considered for the mapping
      dataset.file_to_class = {
          str(idx): dataset[idx][1]
          for idx in range(len(dataset))
          if not dataset.imgs[idx][0].startswith('.')
      }    

      # transform = get_transform(interpolation=InterpolationMode.BICUBIC, size=512)
      # dataset = datasets.ImageFolder(root="/content/drive/MyDrive/shapenet_dataset", transform=transform)

      # # Helper function to extract unique ID from file path
      # def get_unique_id_from_path(file_path):
      #     # Extract the parts of the file path
      #     path_parts = file_path.split(os.sep)
      #     # Print the parts of the path for debugging
      #     print(f"Path parts: {path_parts[-3:-1]}")
      #     # The unique ID is a combination of class name and model folder name
      #     unique_id = "_".join(path_parts[-3:-1])  # Example: 'airplane_model1'
      #     # Print the unique ID for debugging
      #     print(f"Generated unique ID: {unique_id} for file path: {file_path}")
      #     return unique_id
      

      # # Mapping each image to its corresponding unique ID
      # dataset.file_to_class = {}
      # for idx, (path, _) in enumerate(dataset.imgs):
      #     # Extract unique ID from the file path
      #     unique_id = get_unique_id_from_path(path)
      #     dataset.file_to_class[str(idx)] = unique_id
      #     # Print the index and unique ID for debugging
      #     print(f"Index: {idx}, Unique ID: {unique_id}, File path: {path}")

  
    else:
        raise ValueError(f"Dataset {name} not supported.")

    if name in {'mnist', 'cifar10', 'stl10', 'aircraft', 'buildingnet', 'shapenet'}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1]
            for idx in tqdm(range(len(dataset)))
        }

    assert hasattr(dataset, "class_to_idx"), f"Dataset {name} does not have a class_to_idx attribute."
    assert hasattr(dataset, "file_to_class"), f"Dataset {name} does not have a file_to_class attribute."
    return dataset

