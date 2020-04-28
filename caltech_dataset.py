from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def make_dataset(directory, class_to_idx, filename):
    instances = []
    directory = os.path.expanduser(directory)
    
    # SELECT FILE TO READ FOR INPUTS
    input_file_path = os.path.split(directory)[0]
    input_file = input_file_path + "/" + filename
    
    # READ FILE AND IMAGES
    with open(input_file, "r") as file:
        for line in file:
            if line.endswith('\n'):
                line = line.rstrip()
            class_name = line.split("/")[0]
            if (not class_name.startswith("BACKGROUND")):
                class_index = class_to_idx[class_name]
                path = os.path.join(directory, line)
                item = path, class_index
                instances.append(item)
    return instances



class Caltech(VisionDataset):
    def __init__(self, root, split='train.txt', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
                                    # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, self.split)
        
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        
        
    def _find_classes(self, dir):
        '''Finds class folders in the dataset
        
        Args: dir (string): root directory path
        
        Returns: tuple (classes, class_to_idx) where classes relative to (dir) and class_to_idx dictionary
        '''
        
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and not d.name.startswith("BACKGROUND")]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        path, label = self.samples[index]           # Provide a way to access image and label via index
        image = pil_loader(path)                    # Image should be a PIL Image
                                                    # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.samples) # Provide a way to get the length (number of elements) of the dataset
        return length
