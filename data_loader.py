# %%writefile /kaggle/working/UC-GAN-v.2/data_loader.py

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

    dataset = ImageFolder(image_dir, transform) #todo: skip this to have txt compressed to bmp files(that I created manually) instead of have img
    # print("image_dir is??",image_dir)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,

    )
    # print("data_loader is txt or img?????????? ", data_loader) #todo: print text to see whether it is txt or img files
    return data_loader

#dataloader.py => main.py => main.py call Solver_Substi => solver_substi.py