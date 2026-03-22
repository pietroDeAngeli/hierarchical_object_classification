"""
module containing utilities to load
the dataset for the training
of the siamese network on images.
"""
import json
import copy
import functools
import logging
import lz4.frame

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Subset
from skimage import io
import torch

import recsiam.utils as utils

# entry
# { "id" :  int,
#   "paths" : str,
#   "metadata" : str}


def nextid():
    cid = 0
    while True:
        yield cid
        cid += 1


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".npy", ".lz4"}


def descriptor_from_filesystem(root_path):
    """Build a descriptor from a root folder structured as root/class_folder/images."""
    desc = []
    root_path = Path(root_path)
    id_gen = nextid()

    for subd in sorted(root_path.iterdir()):
        if not subd.is_dir():
            continue

        images = sorted(
            str(p) for p in subd.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            continue

        obj_desc = {"id": next(id_gen), "name": str(subd.name), "paths": images}
        desc.append(obj_desc)

    info_path = root_path / "metadata.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    return info, desc


class ImageDataSet(Dataset):
    """
    PyTorch Dataset of images organised as root/class_folder/image_file.
    Indexing: dataset[class_idx, img_idx] -> np.ndarray (C, H, W) or embedding.
    """

    def __init__(self, descriptor):
        self.logger = logging.getLogger("recsiam.data.ImageDataSet")

        self.descriptor = descriptor
        if not isinstance(self.descriptor, (list, tuple, np.ndarray)):
            with Path(self.descriptor).open("r") as ifile:
                self.descriptor = json.load(ifile)

        self.info = self.descriptor[0]
        self.data = np.asarray(self.descriptor[1])

        # paths[class_idx] = list of image paths for that class
        self.paths = np.array([d["paths"] for d in self.data], dtype=object)
        self.img_number = np.array([len(p) for p in self.paths])

        def get_id_entry(elem_id):
            return self.data[elem_id]["id"]
        self.id_table = np.vectorize(get_id_entry)

        # Detect whether files are pre-computed embeddings
        self.embedded = False
        self.compressed = False
        first_path = self.paths[0][0]
        try:
            np.load(first_path)
            self.embedded = True
        except Exception:
            pass
        if not self.embedded:
            try:
                with lz4.frame.open(first_path, mode="rb") as f:
                    np.load(f)
                self.embedded = True
                self.compressed = True
            except Exception:
                pass

        self.n_elems = len(self.paths)

    @property
    def is_embed(self):
        return self.embedded

    def get_metadata(self, key, elem_ind, object_level=True):
        elem_ind = np.asarray(elem_ind)
        obj_d = self.data[elem_ind[0]]
        val = obj_d[key] if object_level else obj_d[key][elem_ind[1]]
        log = logging.getLogger(self.logger.name + ".get_metadata")
        log.debug("key = {}\telem_ind ={}\tval = {}".format(key, elem_ind, val))
        return val

    def load_image(self, path):
        """Load a single image from disk (raw or embedded)."""
        if self.compressed:
            with lz4.frame.open(str(path), mode="rb") as f:
                return np.load(f)
        if self.embedded:
            return np.load(str(path), allow_pickle=True)
        img = io.imread(str(path))
        if img.ndim == 2:          # grayscale -> add channel dim
            img = img[..., None]
        
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float() / 255.0 # Conversion to [0, 1] range for DINO compatibility

    def __len__(self):
        return self.n_elems

    def __getitem__(self, value):
        return self._getitem(value)

    def _getitem(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            if self._valid_t(value):
                return self._get_single_item(*value)
            elif np.all([self._valid_t(v) for v in value]):
                return np.array([self._get_single_item(*v) for v in value])
            else:
                raise TypeError("Invalid argument type: {}.".format(value))
        else:
            raise TypeError("Invalid argument type: {}.".format(value))

    @staticmethod
    def _valid_t(value):
        """Valid index is a 2-tuple (class_idx: int, img_idx: int)."""
        return (
            isinstance(value, (tuple, list, np.ndarray))
            and len(value) == 2
            and isinstance(value[0], (int, np.integer))
            and isinstance(value[1], (int, np.integer))
        )

    def sample_size(self):
        return self._get_single_item(0, 0).shape

    def _get_single_item(self, class_idx, img_idx):
        path = self.paths[class_idx][img_idx]
        log = logging.getLogger(self.logger.name + "._get_single_item")
        log.debug("class_idx={}  img_idx={}  path={}".format(class_idx, img_idx, path))
        return self.load_image(path)

    def gen_embed_dataset(self):
        """Iterate over all (image, path) pairs."""
        for obj in range(self.n_elems):
            for img_idx, path in enumerate(self.paths[obj]):
                yield self[obj, img_idx], path


def dataset_from_filesystem(root_path):
    descriptor = descriptor_from_filesystem(root_path)
    return ImageDataSet(descriptor)


class TrainPairDataSet(ImageDataSet):
    """
    Dataset variant that expects pairs of (class_idx, img_idx) tuples
    and returns the two images together with their class indices.
    """

    def __getitem__(self, value):
        if (
            isinstance(value, (list, tuple, np.ndarray))
            and len(value) == 2
            and np.all([self._valid_t(v) for v in value])
        ):
            items = self._getitem(value)       # shape (2, C, H, W)
            labels = (value[0][0], value[1][0])
            return items, labels
        else:
            raise ValueError(
                "Input must be ((class_idx, img_idx), (class_idx, img_idx)). "
                "Got: {}".format(value)
            )


class FlattenedDataSet(ImageDataSet):
    """
    Flattened view of ImageDataSet: each entry is a single (class_idx, img_idx) pair.
    Supports integer, slice and array indexing.
    """

    def __init__(self, *args, preload=False):
        super().__init__(*args)

        # val_map[i] = (class_idx, img_idx)
        self.val_map = np.array(
            [(cls, img)
             for cls in range(len(self.img_number))
             for img in range(self.img_number[cls])],
            dtype=np.int64,
        )
        self.flen = len(self.val_map)

        self.preloaded = None
        if preload:
            self.preloaded = np.array([self.real_getitem(i) for i in range(self.flen)])

    def map_value(self, value):
        return self.val_map[value]

    def __len__(self):
        return self.flen

    def get_metadata(self, key, elem_ind, **kwargs):
        return super().get_metadata(key, self.val_map[elem_ind].squeeze(), **kwargs)

    def get_label(self, value):
        if isinstance(value, slice):
            return self.map_value(value)[:, 0]
        ndim = np.ndim(value)
        if ndim == 0:
            return int(self.map_value(value)[0])
        elif ndim == 1:
            return self.map_value(value)[:, 0]
        else:
            raise ValueError("np.ndim(value) > 1")

    def __getitem__(self, i):
        if self.preloaded is not None:
            return self.preloaded[i]
        return self.getitems(i)

    def getitems(self, ind):
        if isinstance(ind, slice):
            ind = np.arange(*ind.indices(self.flen))
        if isinstance(ind, (list, tuple, np.ndarray)):
            return np.array([self.real_getitem(i) for i in ind])
        return self.real_getitem(ind)

    def real_getitem(self, value):
        class_idx, img_idx = self.map_value(value)
        image = super().__getitem__((int(class_idx), int(img_idx)))
        return image, int(class_idx)

    def balanced_sample(self, elem_per_class, rnd, separate=False, ind_subset=None):
        if ind_subset is None:
            p_ind = rnd.permutation(len(self.val_map))
        else:
            assert np.unique(ind_subset).size == ind_subset.size
            assert (ind_subset >= 0).all() and (ind_subset < len(self.val_map)).all()
            p_ind = rnd.permutation(ind_subset)

        perm = self.val_map[p_ind]
        cls = perm[:, 0]
        _, indices = np.unique(cls, return_index=True)
        remaining_ind = np.delete(np.arange(len(cls)), indices)
        ind_sets = [indices]

        for _ in range(elem_per_class - 1):
            p = cls[remaining_ind]
            _, ind = np.unique(p, return_index=True)
            ind_sets.append(ind)
            indices = np.concatenate([indices, remaining_ind[ind]])
            remaining_ind = np.delete(remaining_ind, ind)

        if not separate:
            return p_ind[indices]
        return tuple(p_ind[i] for i in ind_sets)

    def get_n_objects(self, number, rnd, ind_subset=None):
        if ind_subset is None:
            elems = len(self.img_number)
        else:
            elems = np.unique(self.get_label(ind_subset))
        obj_ind = rnd.choice(elems, size=number, replace=False)
        class_ind = np.where(np.isin(self.val_map[:, 0], obj_ind))[0]
        if ind_subset is not None:
            class_ind = np.intersect1d(class_ind, ind_subset)
        return class_ind


def list_collate(data):
    # Stack into a single (B, ...) tensor so models receive a proper batch
    emb = torch.stack([utils.astensor(d[0]) for d in data])
    lab = np.array([d[1] for d in data])

    return emb, lab


class ExtendedSubset(Subset):

    def __init__(self, dataset, indices=None):
        if indices is None:
            indices = np.arange(len(dataset))
        if isinstance(dataset, Subset):
            indices = dataset.indices[indices]
            dataset = dataset.dataset

        self.info = dataset.info
        super().__init__(dataset, indices)

    def get_label(self, value):
        return self.dataset.get_label(self.indices[value])

    def get_metadata(self, key, elem_ind, **kwargs):
        return self.dataset.get_metadata(key, self.indices[elem_ind], **kwargs)

    def split_balanced(self, elem_per_class, rnd):
        ind = self.dataset.balanced_sample(elem_per_class, rnd, False, self.indices)

        other_ind = np.setdiff1d(self.indices, ind)

        return (ExtendedSubset(self.dataset, ind),
                ExtendedSubset(self.dataset, other_ind))

    def split_n_objects(self, number, rnd):
        ind = self.dataset.get_n_objects(number, rnd, self.indices)
        other_ind = np.setdiff1d(self.indices, ind)

        return (ExtendedSubset(self.dataset, ind),
                ExtendedSubset(self.dataset, other_ind))

    def same_subset(self, dataset):
        assert len(dataset) == len(self.dataset)
        return ExtendedSubset(dataset, self.indices)



def train_shuf(dataset, seed, dl_args={}, setting=None):
    rs = np.random.RandomState
    rnd_s, rnd_e, rnd_i = [rs(s) for s in rs(seed).randint(2**32 - 1, size=3)]
    if setting is None:
        train_ind = rnd_s.permutation(np.arange(len(dataset)))
    elif setting["type"] == "tree":
        assert "hierarchy" in dataset.info
        names = [dataset.get_metadata("name", i)
                 for i in np.arange(len(dataset))]
        train_ind = utils.shuffle_tree_by_distance(rnd_s,
                                                   dataset.info["hierarchy"],
                                                   names,
                                                   **setting["setting_args"])

    train_ds = ExtendedSubset(dataset, train_ind)
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=False, collate_fn=list_collate, **dl_args)

    return train_dl, None, None


def train_factory(train_desc, test_seed, dl_args={}, ds_args={},
                  setting=None):

    train_ds = FlattenedDataSet(train_desc, **ds_args)

    return functools.partial(train_shuf, train_ds,
                             dl_args=dl_args, setting=setting)
