import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 512), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

        self.transform_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            transforms.ColorJitter(0.3, 0.2, 0.3, 0.1),
            transforms.ToTensor()
        ])

        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        mask = (mask > 0).float()
        return img, mask
