# data/__init__.py
"""
Dataset registry for this repository.
Add new dataset classes here so the training code can look them up by name.
"""

# Keep existing dataset imports (if these modules exist in your repo)
try:
    from .blender import Blender
except Exception:
    Blender = None

try:
    from .llff import LLFF
except Exception:
    LLFF = None

# Import your custom dataset
try:
    from .myvideo import MyVideoDataset
except Exception:
    MyVideoDataset = None

def get_dataset(name):
    """
    Return dataset class matching `name` (case-insensitive).
    """
    if name is None:
        raise ValueError("Dataset name not provided")
    name = name.lower()
    if name == 'blender':
        if Blender is None:
            raise ImportError("Blender dataset not available")
        return Blender
    elif name == 'llff':
        if LLFF is None:
            raise ImportError("LLFF dataset not available")
        return LLFF
    elif name in ('myvideo', 'my_video'):
        if MyVideoDataset is None:
            raise ImportError("MyVideoDataset not available")
        return MyVideoDataset
    else:
        raise NotImplementedError(f"Unknown dataset: {name}")
