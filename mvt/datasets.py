from abc import ABC, abstractmethod
import os

import cv2, json, torch, tqdm, numpy as np


def get_dataset(name: str, root: str, split: str, task: str, transform: callable=None) -> torch.utils.data.Dataset:
    """
    Get a dataset from the dataset zoo.

    Parameters
    ----------
    `name` (str): name of the dataset
    `root` (str): path to the dataset root directory
    `split` (str): split of the dataset (train, val)
    `task` (str): task to perform (classification, detection)
    `transform` (callable): transformation function to apply to the samples

    Return
    ------
    torch.utils.data.Dataset: dataset object
    """

    DATA_ZOO = {
        "mnist":    (MNIST,        {}),
        "cifar10":  (CIFAR,        {"num_classes":  10}),
        "cifar20":  (CIFAR,        {"num_classes":  20}),
        "cifar100": (CIFAR,        {"num_classes": 100}),
        "imagenet": (ILSVRC,       {}),
        "coco":     (COCO,         {}),
        "pets":     (OxfordIITPet, {}),
    }

    assert name in DATA_ZOO, f"Invalid dataset name: {name}"

    dataset_class, kwargs = DATA_ZOO.get(name)

    return dataset_class(root, split, task, transform=transform, **kwargs)


class _Dataset(ABC, torch.utils.data.Dataset):

    def __init__(self, transform: callable=None, *args, **kwargs) -> None:
        """
        Initialize the abstract class dataset.

        Parameters
        ----------
        `*args`: positional arguments
        `**kwargs`: keyword arguments
        """
        super(_Dataset, self).__init__(*args, **kwargs)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        """
        Return a sample from the dataset.

        Parameters
        ----------
        `index` (int): index of the sample

        Return
        ------
        tuple: sample as a tuple of (samples, labels)
        """
        assert index < len(self.data), f"Index out of range: {index}"

        sample = self._load_sample(index)
        labels = self._load_labels(index)

        if self.transform is not None:
            sample, labels = self.transform(sample, labels)

        return sample, labels

    @abstractmethod
    def _load_sample(self, index) -> np.ndarray:
        """
        Load a sample from the dataset.

        Parameters
        ----------
        `index` (int): index of the sample

        Return
        ------
        np.ndarray: sample as a numpy array
        """
        pass

    @abstractmethod
    def _load_labels(self, index: int) -> None:
        """
        Load labels for a sample from the dataset.

        Parameters
        ----------
        `index` (int): index of the sample

        Return
        ------
        int: dataset specific label for the sample
        """
        pass

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """
        Collate a batch of samples. This method is used by the dataloader.

        Parameters
        ----------
        `batch` (list): list of samples

        Return
        ------
        tuple: batch as a tuple of stacked tensors (samples, labels)"""
        samples, labels = zip(*batch)
        return torch.stack(samples, 0), torch.cat(labels, 0)


class MNIST(_Dataset):

    def __init__(self, root: str, split: str, task: str="classification", transform: callable=None, *args, **kwargs) -> None:
        """
        Initialize the MNIST dataset.

        Parameters
        ----------
        `root` (str): path to the dataset root directory
        `split` (str): split of the dataset (train, val)
        `task` (str): task to perform (classification)
        `transform` (callable): transformation function to apply to the samples
        """
        assert os.path.exists(root), f"Dataset directory not found: {root}"
        assert split in ["train", "val"], f"Invalid split: {split}"
        assert task in ["classification"], "MNIST dataset supports only 'classification' task"
        super(MNIST, self).__init__(transform=transform, *args, **kwargs)

        import torchvision
        self.data = torchvision.datasets.MNIST(root, bool(split == "train"), None, None, True)
        self.num_classes = 10

    def _load_sample(self, index: int) -> np.ndarray:
        return np.asarray(self.data[index][0])

    def _load_labels(self, index: int) -> int:
        return int(self.data[index][1])


class CIFAR(_Dataset):

    def __init__(self, root: str, split: str, task: str="classification", transform: callable=None, *args, **kwargs) -> None:
        """
        Initialize the CIFAR dataset.

        Parameters
        ----------
        `root` (str): path to the dataset root directory
        `split` (str): split of the dataset (train, val)
        `task` (str): task to perform (classification)
        `transform` (callable): transformation function to apply to the samples
        """
        self.num_classes = kwargs.pop("num_classes", None)
        assert self.num_classes is not None, "Number of classes not provided for CIFAR dataset"
        assert self.num_classes in [10, 20, 100], "Invalid number of classes for CIFAR dataset. Choose 10, 20 or 100."
        super(CIFAR, self).__init__(transform=transform, *args, **kwargs)

        assert os.path.exists(root), f"Dataset directory not found: {root}"
        assert split in ["train", "val"], f"Invalid split: {split}"
        assert task in ["classification"], "CIFAR dataset currently supports only 'classification' task"

        def _unpickle(file):
            import pickle
            with open(file, "rb") as fo:
                dict = pickle.load(fo, encoding="bytes")
            return dict

        if self.num_classes == 10:
            if split == "train":
                for i in range(1, 6):
                    file_path = os.path.join(root, "cifar", "cifar-10-batches-py", f"data_batch_{i}")
                    file_data = _unpickle(file_path)
                    if i == 1:
                        self.data = file_data[b"data"]
                        self.labels = file_data[b"labels"]
                    else:
                        self.data = np.vstack((self.data, file_data[b"data"]))
                        self.labels += file_data[b"labels"]
            else:
                file_path = os.path.join(root, "cifar", "cifar-10-batches-py", "test_batch")
                file_data = _unpickle(file_path)
                self.data = file_data[b"data"]
                self.labels = file_data[b"labels"]
            self.class_names = [c.decode("utf-8") for c in _unpickle(os.path.join(root, "cifar", "cifar-10-batches-py", "batches.meta"))[b"label_names"]]
        elif self.num_classes in [20, 100]:
            split = "train" if split == "train" else "test"
            file_path = os.path.join(root, "cifar", "cifar-100-python", f"{split}")
            file_data = _unpickle(file_path)
            self.data = file_data[b"data"]
            labels_key, label_names_key = (b"coarse_labels", b"coarse_label_names") if self.num_classes == 20 else (b"fine_labels", b"fine_label_names")
            self.labels = file_data[labels_key]
            self.class_names = [c.decode("utf-8") for c in _unpickle(os.path.join(root, "cifar", "cifar-100-python", "meta"))[label_names_key]]

        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def _load_sample(self, index: int) -> np.ndarray:
        return self.data[index]

    def _load_labels(self, index: int) -> int:
        return self.labels[index]


class COCO(_Dataset):

    def __init__(self, root: str, split: str, task: str="detection", transform: callable=None, *args, **kwargs):
        """
        Initialize the COCO dataset.
        
        Parameters
        ----------
        `root` (str): path to the dataset root directory
        `split` (str): split of the dataset (train, val)
        `task` (str): task to perform (detection, segmentation)
        `transform` (callable): transformation function to apply to the samples
        """
        super(COCO, self).__init__(transform=transform, *args, **kwargs)

        # sanity checks
        assert os.path.exists(root), f"Dataset directory not found: {root}"
        assert split in ["train", "val"], f"Invalid split: {split}"
        assert task=="detection", "COCO dataset currently supports only 'detection' task"

        file_path = os.path.join(root, "coco", "annotations", f"instances_{split}2017.json")
        print(f"Loading COCO annotations: {file_path}")
        file_data = json.load(open(file_path, "r"))

        # create a dictionary of annotations to map image ids to their annotations
        ann_dict = {}
        for ann_data in file_data["annotations"]:
            image_id = ann_data["image_id"]
            if image_id not in ann_dict:
                ann_dict[image_id] = []
            ann_dict[image_id].append(ann_data)

        self.data = []
        class_ids = set()

        print("Parsing COCO annotations")
        for img_data in tqdm.tqdm(file_data["images"]):
            image_path = os.path.join(root, "coco", "images", f"{split}2017", img_data["file_name"])
            image_id = img_data["id"]
            boxes = []
            if image_id in ann_dict:
                for ann_data in ann_dict[image_id]:
                    x, y, w, h = [int(v) for v in ann_data["bbox"]]
                    cls = int(ann_data["category_id"])
                    boxes.append([x, y, x+w, y+h, cls])
                    class_ids.add(cls)
            self.data.append([image_path, boxes])

        self.max_boxes_per_image = max([len(d[1]) for d in self.data])
        self.num_classes = len(class_ids)

    def _load_sample(self, index: int) -> np.ndarray:
        sample = cv2.imread(self.data[index][0], -1)
        if len(sample.shape) == 2: sample = np.stack([sample] * 3, -1)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        return sample

    def _load_labels(self, index: int) -> int:
        labels = self.data[index][1]
        # pad the boxes with zeros to make them of equal length
        while len(labels) < self.max_boxes_per_image:
            labels.append([0, 0, 0, 0, 0])
        return np.asarray(labels, dtype=np.float32)


class ILSVRC(_Dataset):

    def __init__(self, root: str, split: str, task: str="classification", transform: callable=None, *args, **kwargs):
        """
        Initialize the ILSVRC dataset.
        
        Parameters
        ----------
        `root` (str): path to the dataset root directory
        `split` (str): split of the dataset (train, val)
        `task` (str): task to perform (classification, detection)
        `transform` (callable): transformation function to apply to the samples
        """
        super(ILSVRC, self).__init__(transform=transform, *args, **kwargs)

        assert os.path.exists(root), f"Dataset directory not found: {root}"
        assert split in ["train", "val"], f"Invalid split: {split}"
        assert task in ["classification", "detection"] , "ILSVRC dataset currently supports only 'classification' and 'detection' tasks"

        file_path = os.path.join(root, "ILSVRC", "LOC_synset_mapping.txt")
        print(f"Loading ILSVRC synset mapping: {file_path}")
        with open(file_path, "r") as fp:
            file_data = fp.readlines()

        synset_to_class_id = {}
        counter = 0

        for line in tqdm.tqdm(file_data):
            synset = line.split()[0]
            synset_to_class_id.update({synset: counter})
            counter += 1

        self.num_classes = len(synset_to_class_id)
        self.max_boxes_per_image = 0

        file_path = os.path.join(root, "ILSVRC", f"LOC_{split}_solution.csv")
        print(f"Loading ILSVRC annotations: {file_path}")
        with open(file_path, "r") as fp:
            file_data = fp.readlines()[1:]

        self.data = []

        print("Parsing ILSVRC annotations")
        for line in tqdm.tqdm(file_data):
            image_name, annotations = line.split(",")
            class_dir = image_name.split("_")[0] if split == "train" else ""
            image_path = os.path.join(root, "ILSVRC", "Data", "CLS-LOC", split, class_dir, image_name + ".JPEG")
            annotations = annotations.split()

            if task == "detection":
                boxes = []
                for i in range(len(annotations) // 5):
                    synset = annotations[i*5]
                    cls = synset_to_class_id[synset]
                    x0, y0, x1, y1 = [int(v) for v in annotations[i*5+1:i*5+5]]
                    boxes.append([x0, y0, x1, y1, cls])
                self.data.append([image_path, boxes])
                self.max_boxes_per_image = max([len(d[1]) for d in self.data])
            elif task == "classification":
                synset = annotations[0]
                cls = synset_to_class_id[synset]
                self.data.append([image_path, cls])

    def _load_sample(self, index: int) -> np.ndarray:
        sample = cv2.imread(self.data[index][0], -1)
        if len(sample.shape) == 2: sample = np.stack([sample] * 3, -1)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        return sample

    def _load_labels(self, index: int) -> int:
        labels = self.data[index][1]
        dtype = np.int64  # default dtype for classification

        # if the task is detection, pad the boxes with zeros to make them of equal length
        if isinstance(labels, list):
            dtype = np.float32  # dtype for detection
            while len(labels) < self.max_boxes_per_image:
                labels.append([0, 0, 0, 0, 0])

        return np.asarray(labels, dtype)


class OxfordIITPet(_Dataset):

    def __init__(self, root: str, split: str, task: str="classification", transform: callable=None, *args, **kwargs) -> None:
        """
        Initialize the Oxford IIIT Pet dataset.
        
        Parameters
        ----------
        `root` (str): path to the dataset root directory
        `split` (str): split of the dataset (train, val)
        `task` (str): task to perform (classification, detection, segmentation)
        """

        super(OxfordIITPet, self).__init__(transform=transform, *args, **kwargs)

        assert os.path.exists(root), f"Dataset directory not found: {root}"
        assert split in ["train", "val"], f"Invalid split: {split}"
        assert task in ["classification"], "Oxford IIIT Pet dataset currently supports only 'classification' task"

        file_path = os.path.join(root, "oxford_iiit_pet", "annotations", "trainval.txt" if split == "train" else "test.txt")
        print(f"Loading Oxford IIIT Pet annotations: {file_path}")
        with open(file_path, "r") as fp:
            file_data = fp.readlines()

        self.data = []

        print("Parsing Oxford IIIT Pet annotations")
        for line in tqdm.tqdm(file_data):
            image_name, class_id, specie, breed_id = line.split()
            image_path = os.path.join(root, "oxford_iiit_pet", "images", f"{image_name}.jpg")
            if task == "classification":
                self.data.append([image_path, int(specie) - 1])

        self.num_classes = len(set([d[1] for d in self.data]))

    def _load_sample(self, index: int) -> np.ndarray:
        sample = cv2.imread(self.data[index][0], -1)
        assert sample is not None, f"Image not found: {self.data[index][0]}"
        if len(sample.shape) == 2: sample = np.stack([sample] * 3, -1)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        return sample

    def _load_labels(self, index: int) -> int:
        assert self.data[index][1] < self.num_classes, f"Invalid class id: {self.data[index][1]}"
        return self.data[index][1]