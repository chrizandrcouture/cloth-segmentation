import os
import pdb

from tqdm import tqdm
import PIL
from PIL import Image

import numpy as np
import cv2

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET


class SegmentImages:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = self.load_model(checkpoint_path, device)
        self.class_labels = {"background": 0, "upper": 1, "lower": 2, "full": 3}
        self.transform = self.get_transforms()

    def load_model(self, checkpoint, device):
        net = U2NET(in_ch=3, out_ch=4)
        net = load_checkpoint_mgpu(net, checkpoint_path)
        net = net.to(device)
        net = net.eval()
        return net

    def get_transforms(self):
        transforms_list = []
        transforms_list += [transforms.ToTensor()]
        transforms_list += [Normalize_image(0.5, 0.5)]
        transform_rgb = transforms.Compose(transforms_list)
        return transform_rgb

    def find_corners(self, indices):
        x, y = indices
        top_left = x.min(), y.min()
        bottom_right = x.max(), y.max()
        return top_left, bottom_right

    def get_img_mask(self, image_name, label):
        try:
            img = Image.open(image_name).convert("RGB")
        except PIL.UnidentifiedImageError:
            print(f"[ERROR]:  Corrupted image at {image_name}")
            return None, None

        img_arr = np.array(img)
        image_tensor = self.transform(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        output_tensor = self.model(image_tensor.to(self.device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

        return output_arr, img_arr

    def get_segmented_region(self, mask, img_arr, image_name, label):
        indices = (mask == self.class_labels[label]).nonzero()
        if len(indices[0]) != 0:
            (x1, y1), (x2, y2) = self.find_corners(indices)
            output_img = np.ones((img_arr.shape), dtype=np.uint8) * 255
            output_img[indices] = img_arr[indices]
            output_img = output_img[x1: x2, y1: y2]
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            return output_img
        print(f"[ERROR]:  No {label} mask found in image {image_name}")
        return None

    def process_folder(self, image_folder, label, ext, output_ext):
        images_list = [os.path.join(images_folder, x) for x in os.listdir(images_folder)
                       if not x.endswith(output_ext) and x.endswith(ext)]

        for image_name in tqdm(images_list):
            mask, img_arr = self.get_img_mask(image_name, label)
            if mask is None:
                continue

            segmented_img = self.get_segmented_region(mask, img_arr, image_name, label)
            if segmented_img is None:
                continue

            try:
                output_path = image_name.replace(ext, output_ext)
                cv2.imwrite(output_path, output_img)
            except cv2.error:
                print(f"[ERROR]:  Could not write {image_name}, corrupted pixel values")


def segment_ajio_catalogue():
    TOPS = [
        "Men/WesternWear/JacketsCoats",
        "Men/WesternWear/Shirts",
        "Men/WesternWear/Tshirts",
        "Women/WesternWear/JacketsCoats",
        "Women/WesternWear/Shirts",
        "Women/WesternWear/Tshirts",
        "Women/WesternWear/Tops",
    ]

    FULL = ["Women/WesternWear/Dresses"]

    BOTTOMS = [
        "Women/WesternWear/JeansJeggings",
        "Women/WesternWear/Skirts",
        "Women/WesternWear/Shorts",
        "Women/WesternWear/TrousersPants",
        "Women/WesternWear/TrackPants",
        "Men/WesternWear/Jeans",
        "Men/WesternWear/TrackPants",
        "Men/WesternWear/Shortsths",
        "Men/WesternWear/TrousersPants",
    ]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "cloth_segm_u2net_latest.pth"
    EXT = ".jpg"
    OUTPUT_EXT = "-segmented.jpg"
    # image_dir = "/data/chris/cross-dress/ajio-large/imagesfortwol1l2"
    IMAGE_DIR = "/data/chris/cross-dress/ajio-data/AjioCatalogueRand50Imgs"
    # result_dir = "/data/chris/cross-dress/ajio-large/imagesfortwol1l2"
    result_dir = "/data/chris/cross-dress/ajio-data/AjioCatalogueRand50Imgs"


    extractor = SegmentImages(CHECKPOINT, DEVICE)

    for segment, label in zip([TOPS, FULL, BOTTOMS], ["upper", "full", "lower"]):
        for folder in segment:
            images_folder = os.path.join(image_dir, folder)
            extractor.process_folder(image_folder, label, EXT, OUTPUT_EXT)


if __name__ == "__main__":
    segment_ajio_catalogue()