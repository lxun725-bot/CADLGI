"""Paper preprocessing: bilinear aspect-preserving resize and black padding."""
from PIL import Image, ImageOps
from torchvision import transforms as T

class SmartResize:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        nw = max(1, min(self.target_size, round(w * scale)))
        nh = max(1, min(self.target_size, round(h * scale)))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        left, top = (self.target_size - nw) // 2, (self.target_size - nh) // 2
        return ImageOps.expand(img, (left, top, self.target_size - nw - left,
                                    self.target_size - nh - top), fill=0)

def get_transforms(split, size):
    return _build(split, size, strong=False)

def get_transforms_strong_augmentation(split, size):
    return _build(split, size, strong=True)

def _build(split, size, strong):
    ops = [SmartResize(size)]
    if split == 'train':
        if strong:
            ops.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET,
                                    interpolation=T.InterpolationMode.BILINEAR))
        ops.extend([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.3)])
    ops.append(T.ToTensor())
    if split == 'train' and strong:
        # Random region, black fill before normalization; preserve the original fill convention.
        ops.append(T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0))
    ops.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(ops)
