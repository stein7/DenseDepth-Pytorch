import os
import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
"""
ReadMe
커스텀 dataset과 dataloader 오브젝트를 사용하는 예제코드(실행은 X)
"""
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, fname) 
                            for fname in os.listdir(directory) 
                            if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure RGB
        if self.transform:
            image = self.transform(image)
        return image

def extract_zip(input_zip, output_dir):
    with zipfile.ZipFile(input_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

# Parameters
zip_path = '/home/sslunder0/project/NNPROJ/DenseDepth-Pytorch/densedepth/examples/images.zip'  # Change this to the path of your zip file
extracted_dir = 'extracted_images'

# Step 1: Extract Zip
extract_zip(zip_path, extracted_dir)

# Step 2: Define Transformations and Dataset
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = ImageDataset(directory=extracted_dir, transform=transformations)

# Step 3: Create DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Example of using DataLoader (e.g., in a training loop)
for images in dataloader:
    # Process your images here
    # For example, pass them through a model
    pass
