from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


def get_loader(
    image_dir,
    batch_size=16,
    mode="train",
    num_workers=1,
):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Grayscale()) #compress RGB
    transform.append(T.ToTensor())   # Normalize from [0, 255] to [0, 1]
    transform = T.Compose(transform)

    dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,

    )
    return data_loader

#    transform.append(T.ToPILImage())