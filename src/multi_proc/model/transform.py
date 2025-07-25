from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, ToPILImage,  Normalize, InterpolationMode, RandomResizedCrop, GaussianBlur


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_pixels):
    # rotation_angles = [0, 90, 180, 270]
    # random_int = torch.randint(0, 4, (1,))[0].item()
    # angle = rotation_angles[random_int]
    return Compose([
        Resize(n_pixels, interpolation = InterpolationMode.BICUBIC),
        # RandomResizedCrop(n_pixels, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_pixels),
        # GaussianBlur(3),
        _convert_image_to_rgb,
        # transforms.RandomRotation((angle, angle)),
        ToTensor(),
        
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ToPILImage()
    ])