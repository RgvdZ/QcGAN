import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeblurDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, is_train=True):
        """
        Expects a directory structure:
        root_dir/
            blur/
            sharp/
        """
        self.root_dir = root_dir
        self.blur_dir = os.path.join(root_dir, 'blur')
        self.sharp_dir = os.path.join(root_dir, 'sharp')
        self.image_names = sorted(os.listdir(self.blur_dir))

        # The model takes image patches of 256x256 as input [cite: 122]
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        blur_img_path = os.path.join(self.blur_dir, img_name)
        sharp_img_path = os.path.join(self.sharp_dir, img_name)

        blur_img = Image.open(blur_img_path).convert('RGB')
        sharp_img = Image.open(sharp_img_path).convert('RGB')

        # To ensure same crop for both images, we combine them, transform, and split
        w, h = blur_img.size
        combined = Image.new('RGB', (w * 2, h))
        combined.paste(blur_img, (0, 0))
        combined.paste(sharp_img, (w, 0))

        # Apply deterministic crop via seeding or manual cropping for pairs
        # For simplicity in this implementation, we apply a random crop state manually
        seed = torch.random.seed()
        torch.manual_seed(seed)
        blur_tensor = self.transform(blur_img)
        torch.manual_seed(seed)
        sharp_tensor = self.transform(sharp_img)

        return blur_tensor, sharp_tensor
