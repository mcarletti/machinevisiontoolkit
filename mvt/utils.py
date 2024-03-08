from argparse import Namespace
from collections.abc import Iterable
import os, random

import cv2, yaml, numpy as np
import torch
from torch.backends import cudnn


def string_to_shape(s: str) -> tuple:
    """
    Convert a string to a shape tuple.

    Parameters
    ----------
    `s` (str): input string in the format `HxW` or `CxHxW`

    Return
    ------
    tuple: shape tuple in the format `(1, H, W)` or `(C, H, W)`
    """
    # sanity checks
    numbers = s.split("x")
    assert sum([v.isdigit() for v in numbers]) == len(numbers), f"Invalid shape string. Found non-integer values: {numbers}"
    assert len(numbers) in [2, 3], f"Invalid shape string. Found {len(numbers)} numbers, expected 2 or 3"
    shape = tuple(int(v) for v in numbers)
    if len(shape) == 2:
        shape = (1, *shape)
    return shape


def load_config(filepath: str) -> Namespace:
    """
    Load a YAML file and return a Namespace object.
    
    Parameters
    ----------
    `filepath` (str): path to the YAML file
    
    Return
    ------
    Namespace: configuration object
    """
    # sanity checks
    assert filepath.endswith((".yml", ".yaml")), "Invalid file extension. It must be a YAML (.yml or .yaml) file)"
    assert os.path.exists(filepath), f"File not found: {filepath}"
    with open(filepath, "r") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return Namespace(**config)


def to_list(x: Iterable, dtype=None) -> list:
    """
    Convert an iterable object to a list.
    
    Parameters
    ----------
    `x` (Iterable): input object
    `dtype` (type): data type of the elements of the list; if None, the elements are not converted
    
    Return
    ------
    list: converted object
    """
    if isinstance(x, Iterable):
        L = list(x) if not isinstance(x, list) else x
    else:
        L = [x]
    return [dtype(l) for l in L] if dtype else L


def intersect_dicts(da: dict, db: dict, exclude: tuple=()) -> dict:
    """
    Dictionary intersection of matching keys and shapes, omitting 'exclude' keys

    Returns
    -------
    (dict):  dictionary of intersected matching keys and shapes.
    """
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def draw_box_on_image(image: np.ndarray, xyxy: Iterable, color: Iterable=(255, 0, 0), thickness: int=2) -> None:
    """
    Draw a rectangle on an image.
    
    Parameters
    ----------
    `image` (np.ndarray): input image in BGR format
    `xyxy` (Iterable): coordinates of the rectangle in xyxy format
    `color` (Iterable): color of the rectangle in BGR format
    `thickness` (int): thickness of the rectangle
    
    Return
    ------
    None
    """
    # sanity checks
    assert len(image.shape) == 3 and image.shape[2] == 3, "Invalid image shape. It must be in BGR format"
    assert len(xyxy) == 4, f"Invalid number of box coordinates. Found {len(xyxy)}, expected 4"
    assert xyxy[0] in range(0, image.shape[1]) and xyxy[2] in range(0, image.shape[1]), "Invalid x coordinates"
    assert xyxy[1] in range(0, image.shape[0]) and xyxy[3] in range(0, image.shape[0]), "Invalid y coordinates"
    assert len(color) == 3, f"Invalid color. Received {color}, expected BGR format"
    # make sure the coordinates are positive integers
    box = np.array(xyxy, dtype=np.int32)
    # draw a rectangle on the image
    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, thickness)


def draw_text_on_image(image: np.ndarray, text: str, xy: Iterable, color: Iterable=(255, 0, 0), font_scale: float=1, thickness: int=2) -> None:
    """
    Draw text on an image.
    
    Parameters
    ----------
    `image` (np.ndarray): input image in BGR format
    `text` (str): text to draw
    `xy` (Iterable): coordinates of the text
    `color` (Iterable): color of the text in BGR format
    `font_scale` (float): font scale
    `thickness` (int): thickness of the text
    
    Return
    ------
    None
    """
    # sanity checks
    assert len(image.shape) == 3 and image.shape[2] == 3, "Invalid image shape. It must be in BGR format"
    assert len(xy) == 2, f"Invalid number of text coordinates. Found {len(xy)}, expected 2"
    assert xy[0] in range(0, image.shape[1]) and xy[1] in range(0, image.shape[0]), "Invalid x, y coordinates"
    assert len(color) == 3, f"Invalid color. Received {color}, expected BGR format"
    # draw text on the image
    cv2.putText(image, text, tuple(xy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def show_image(image: np.ndarray, window_name: str="Image", delay: int=0) -> None:
    """
    Show an image in a window.
    
    Parameters
    ----------
    `image` (np.ndarray): input image in RGB or grayscale format
    `window_name` (str): name of the window
    `delay` (int): time to wait for a key press in milliseconds
    
    Return
    ------
    None
    """
    # sanity checks
    assert (len(image.shape) == 3 and image.shape[2] == 3) or len(image.shape) == 2, "Invalid image shape. It must be in RGB or grayscale format"
    # show the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
    cv2.imshow(window_name, image)
    cv2.waitKey(delay)


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    `seed` (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_rng_state() -> dict:
    """
    Get the state of the random number generators.

    Return
    ------
    dict: state of the random number generators
    """
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "hash": os.environ["PYTHONHASHSEED"],
    }


def set_rng_state(state: dict) -> None:
    """
    Set the state of the random number generators.

    Parameters
    ----------
    `state` (dict): state of the random number generators
    """
    if not all(k in state for k in ["random", "numpy", "torch", "cuda", "hash"]):
        print("Invalid random generators state dictionary. It must contain the keys 'random', 'numpy', 'torch', 'cuda', and 'hash'")
        return

    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    os.environ["PYTHONHASHSEED"] = state["hash"]


def set_reproducibility(seed: int, rng_state: dict={}, gpu_determinism: bool=False) -> None:
    """
    Enable reproducibility on operations related to tensor and randomness.

    Parameters
    ----------
    `seed` (int): random seed
    `deterministic` (bool): enable deterministic operations
    `rng_state` (dict): state of the random number generators

    References
    ----------
    https://numpy.org/doc/stable/reference/random/generator.html
    https://github.com/ultralytics/yolov5/pull/8213
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    """

    set_seed(seed)
    set_rng_state(rng_state)

    # let or prevent cudnn to autonomously select the underlying algorithm according to the operation
    cudnn.benchmark = not gpu_determinism

    # ask pytorch to use deterministic algorithms, if available;
    # raise an error if a nondeterministic operation is called
    torch.use_deterministic_algorithms(gpu_determinism)
    cudnn.deterministic = gpu_determinism

    # set a debug environment variable
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"