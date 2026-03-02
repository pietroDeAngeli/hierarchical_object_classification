import torch
import logging
import torch.nn.functional as F
import collections
import numpy as np
import sys
from .utils import default_image_normalizer, t2a
from . import utils
from . import memory as mem


def compute_embed_shape(net, shape):
    """ cmpute shape in the resulting feature embedding
    given an input of size shape [C,H,W] """

    tensor_example = torch.zeros((1,) + tuple(shape))

    forwarded = net.forward(tensor_example)

    unbatched_size = tuple(forwarded.shape)[1:]

    return unbatched_size


def flatten_batch(batch):

    flattened_shape = tuple(batch.shape[0:1]) + \
                            (int(np.prod(batch.shape[1:])),)
    flattened = batch.reshape(flattened_shape)
    return flattened


class Noise(torch.nn.Module):
    def forward(self, data):
        assert type(data) in (list, tuple)
        ret = []
        for x in data:
            rand = np.random.uniform(-0.1, 0.1, size=x.shape).astype(np.float32)
            ret.append(x + rand)
        return ret


class BatchFlattener(torch.nn.Module):

    def forward(self, batch):
        return flatten_batch(batch)


class RunningMean(torch.nn.Module):
    def __init__(self, window_size, stride=1):
        super().__init__()
        self.window_size = window_size
        self.stride = stride

        self._c_window = None
        self._c_stride = None

    def forward(self, batch):

        return [self.forward_single(b) for b in batch]

    def forward_single(self, data):
        if self.window_size == 1 and self.stride == 1:
            return data

        elif self.window_size == -1:
            if data.shape[0] == 1:
                return data
            else:
                return data.mean(dim=0)[None, ...]

        else:
            if self._c_window is None or self._c_stride is None:
                self._c_window = torch.ones(1, 1, self.window_size, 1) / self.window_size
                self._c_stride = (self.stride, 1)

            ravg = torch.nn.functional.conv2d(data[None, None, ...],
                                              self._c_window,
                                              stride=self._c_stride)

            return ravg[0, 0, ...]


def EmbeddingTransformer(weights, multiply=False):
    if isinstance(weights, (tuple, list)):
        lin = torch.nn.Linear(weights[1], weights[0], bias=False)

    else:
        lin = torch.nn.Linear(weights.shape[1], weights.shape[0], bias=False)
        if multiply:
            lin.weight.data[...] *= weights
        else:
            lin.weight.data[...] = weights

    return SequenceSequential(lin)


def EmbeddingReductor(elements, factor, bias=False):
    return SequenceSequential(EmbeddingGroupReductor(elements, factor, bias=bias))


class EmbeddingGroupReductor(torch.nn.Module):

    def __init__(self, elements, factor, bias=False):
        super().__init__()
        assert elements % factor == 0

        div = int(elements/factor)
        self.c = torch.nn.Conv1d(elements, div, 1, groups=div, bias=bias)

    def forward(self, data):
        return self.c(data[:, :, None])[:, :, 0]


def EmbeddingMeanReductor(elements, factor):
    assert elements % factor == 0

    r_size = int(elements / factor)
    base_w = np.tile(np.eye(r_size), factor).reshape((r_size * factor, r_size))
    base_w /= float(factor)

    return EmbeddingTransformer(utils.a2t(base_w.T).float())


def GlobalMean():
    return RunningMean(-1)


def ReducedMean(shape, factor):
    return torch.nn.Sequential(EmbeddingMeanReductor(shape, factor), GlobalMean())


class MultiRunningMean(torch.nn.Module):
    def __init__(self, window_sizes, strides):
        assert len(window_sizes) == len(strides)

        super().__init__()
        self.window_sizes = window_sizes
        self.strides = strides

        self.running_means = [RunningMean(w, s) for w, s in zip(self.window_sizes, self.strides)]

    def forward(self, data):
        results = [r_m.forward(data) for r_m in self.running_means]
        return [torch.cat(list(r)) for r in zip(*results)]


class RecursiveReduction(torch.nn.Module):
    def __init__(self, elements, window_size=2, stride=2,
                 activation=torch.nn.functional.leaky_relu, 
                 init_avg=False, output_info=False):
        super().__init__()

        self.window_size = window_size
        self.stride = stride

        self.info = output_info
        self.activation = activation

        self.c = None
        self.init_conv(elements, init_avg)

    def forward(self, batch, info=True):
        return [self.forward_single(b) for b in batch]

    def forward_single(self, data):
        data  = data[None, None, ...].permute(0, 3, 1, 2)
        # i need to finish this
        rec_info = []
        while data.shape[-1] > 1:
            to_pad = self.stride - (data.shape[-1] - self.window_size) % self.stride
            rec_info.append(to_pad % self.stride)

            if np.any(to_pad < self.stride):
                l_pad = np.floor(to_pad / 2).astype(int)
                u_pad = np.ceil(to_pad / 2).astype(int)

                data = F.pad(data, (l_pad, u_pad, 0, 0), mode="replicate")
            data = self.activation(self.c(data))
        
        if self.info:
            return data.squeeze()[None, ...], rec_info
        else:
            return data.squeeze()[None, ...]

    def init_conv(self, elements, init_avg):
        if self.c is None:
            w = torch.eye(elements)[None, ...].repeat((self.window_size,1,1)) 
            w = (w / self.window_size)[None, ...].permute(2,3,0,1)
            self.c = torch.nn.Conv2d(elements, 
                                     elements,
                                     (1, self.window_size),
                                     stride=self.stride, bias=True)

            if init_avg:
                c_w = self.c._parameters["weight"]

                self.c._parameters["weight"] = torch.nn.Parameter( c_w *1e-1 +  w)


class RecursiveExpansion(torch.nn.Module):
    def __init__(self, elements, window_size=2, stride=2,
                 activation=torch.nn.functional.leaky_relu, 
                 init_avg=False):
        super().__init__()

        self.window_size = window_size
        self.stride = stride

        self.activation = activation

        self.c = None
        self.init_conv(elements, init_avg)

    def forward(self, batch):
        return [self.forward_single(b[0], b[1]) for b in batch]

    def forward_single(self, data, rec_info):
        assert isinstance(data, torch.Tensor)
        assert isinstance(rec_info, list)

        data  = data[None, None, ...].permute(0, 3, 1, 2)
        # i need to finish this
        for to_pad in reversed(rec_info):
            data = self.activation(self.c(data))

            if to_pad > 0:
                l_pad = np.floor(to_pad / 2).astype(int)
                u_pad = np.ceil(to_pad / 2).astype(int)

                data = data[:, :, :, l_pad:-u_pad]
        
        return data.squeeze().permute(1,0)

    def init_conv(self, elements, init_avg):
        if self.c is None:
            w = torch.eye(elements)[None, ...].repeat((self.window_size,1,1)) 
            w = (w / self.window_size)[None, ...].permute(2,3,0,1)
            self.c = torch.nn.ConvTranspose2d(elements, 
                                     elements,
                                     (1, self.window_size),
                                     stride=self.stride, bias=True)

            if init_avg:
                raise NotImplementedError()
                c_w = self.c._parameters["weight"]

                self.c._parameters["weight"] = torch.nn.Parameter( c_w *1e-1 +  w)


class RecursiveAutoencoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):

        super().__init__()
        self.enc = RecursiveReduction(*args, **kwargs)
        self.dec = RecursiveExpansion(*args, **kwargs)

    def forward(self, data):
        self.enc.info = self.training
        encoded = self.enc(data)

        if not self.training:
            return encoded
        else:
            decoded = self.dec(encoded)
            return decoded


class ConvolutionalTransform(torch.nn.Module):
    def __init__(self, elements, window_size=(8, 256), stride=(1, 1),
                 activation=torch.nn.functional.leaky_relu, 
                 init_avg=False):
        super().__init__()

        self.window_size = np.asarray(window_size)
        assert self.window_size.size == 2

        self.stride = np.asarray(stride)
        assert self.stride.size == 2

        self.activation = activation

        c_number = np.ceil(elements / int(self.window_size[1])).astype(int)
        c_list = [torch.nn.Conv2d(1,
                                  1,
                                  tuple(self.window_size),
                                  stride=tuple(self.stride), bias=True)
                       for _ in range(c_number)]
        self.c_list = torch.nn.Sequential(*c_list)


    def forward(self, batch):
        return [self.forward_single(b) for b in batch]

    def forward_single(self, data):
        d_shape = np.asarray(data.shape)
        data = data[None, None, ...]
        to_pad = self.get_correct_padding(d_shape)
        l_pad, u_pad = get_pads(to_pad)
        fwded = F.pad(data, (l_pad[1], u_pad[1], l_pad[0], u_pad[0]), mode="constant")

        for conv in self.c_list:
            fwded = self.activation(conv(fwded))

        assert data.shape == fwded.shape
        return fwded[0, 0, ...]

    def get_correct_padding(self, dims):
        dims = np.asarray(dims)
        assert dims.size == 2

        def _get_pad(input_dim):
            return input_dim * (self.stride - 1) + (self.window_size ) - self.stride

        padding = 0
        for conv in self.c_list:
            padding +=_get_pad(dims + padding)

        return padding

def get_pads(to_pad):
    l_pad = np.floor(to_pad / 2).astype(int)
    u_pad = np.ceil(to_pad / 2).astype(int)
    return l_pad, u_pad


def ConvolutionalTransformMean(*args, **kwargs):
    ct = ConvolutionalTransform(*args, **kwargs)
    mn = GlobalMean()

    return torch.nn.Sequential(ct, mn)


def sigmoid(x, factor, shift):
    return 1.9 / (1 + np.exp(x*factor - shift))  + 0.4


class DistanceAggregation(torch.nn.Module):

    def __init__(self, avg_rep=False, overlapping=False,
                 f_factor=1.0, f_discount=1.0,
                 b_factor=1.0, b_discount=1.0):
        super().__init__()

        assert not (avg_rep and overlapping)
        self.avg_rep = avg_rep
        self.overlapping = overlapping

        max_frames = 1000

#        self.f_factors = f_factor * f_discount**np.arange(1, max_frames + 1)
#        self.b_factors = b_factor * b_discount**np.arange(1, max_frames + 1)

        self.f_factors = sigmoid(np.arange(max_frames), f_factor, f_discount)
        self.b_factors = sigmoid(np.arange(max_frames), b_factor, b_discount)

        logger = logging.getLogger("recsiam.models.DistanceAggregation")
        logger.debug("first {} f_factors = {}".format(20, self.f_factors[:20].round(2)))
        logger.debug("first {} b_factors = {}".format(20, self.b_factors[:20].round(2)))

    
    @torch.jit.ignore
    def forward(self, batch, **kwargs):
        assert "agent" in kwargs
        threshold = kwargs["agent"].linear_threshold
        return [self.forward_single(b, threshold) for b in batch], kwargs

    def forward_single(self, data, threshold):

        logger = logging.getLogger("recsiam.models.DistanceAggregation")
        itx = enumerate(iter(data))
        ind, rep = next(itx)

        f_i = 0
        b_i = 0

        rep_list = []
        for i, frame in itx:
            if torch.norm(rep - frame) > threshold * self.f_factors[f_i]:
                rep_list.append(data[ind:i].mean(dim=0))
                logger.debug("added visobj from frame {} to {}".format(ind, i))

                f_i = 0
                if not self.overlapping:
                    ind = i
                    rep = frame

                else:
                    b_i = 0
                    while torch.norm(data[ind] - frame) > threshold / self.b_factors[b_i]:
                        ind += 1
                        rep = data[ind]

                    assert ind <= i

            else:
                f_i += 1
                if self.avg_rep:
                    rep = data[ind:i].mean(dim=0)

        if ind  < len(data) and False:
            rep_list.append(data[ind:].mean(dim=0))

        logger.debug("number of embeddings = {}".format(len(rep_list)))
        return torch.stack(rep_list)

class FastDistanceAggregation(DistanceAggregation):

    def forward_single(self, data, threshold):

        logger = logging.getLogger("recsiam.models.DistanceAggregation")
        logger.debug("data.shape = {}".format(data.shape))
        dist_mat = mem.cart_euclidean_using_matmul(data, data)

        ind = 0
        f_i = 0
        b_i = 0

        rep_list = []
        i = 1
        while i < len(data):
            if dist_mat[ind, i] > threshold * self.f_factors[f_i]:
                rep_list.append(data[ind:i].mean(dim=0))
                logger.debug("added visobj from frame {} to {}".format(ind, i))

                f_i = 0

                if not self.overlapping:
                    ind = i

                else:
                    b_i = 0
                    ind += 1
                    while dist_mat[ind, i] > threshold / self.b_factors[b_i]:
                        ind += 1
                        b_i +=1
                    assert ind <= i
                    i = ind
            else:
                f_i += 1

            i +=1

        if ind < len(data):
            rep_list.append(data[ind:].mean(dim=0))
            logger.debug("added visobj from frame {} to {}".format(ind, len(data)))

        logger.debug("number of embeddings = {}".format(len(rep_list)))
        return torch.stack(rep_list)

class SequenceSequential(torch.nn.Sequential):

    def forward(self, data):

        i_len = [len(item) for item in data]
        data = torch.cat(data)
        data = super().forward(data)

        data = torch.split(data, i_len)

        return data


class KwargsSequential(torch.nn.Sequential):

    def forward(self, inputs, **kwargs):
        for module in self._modules.values():
            inputs, kwargs = module(inputs, **kwargs)
        return inputs, kwargs


class ParamForward(torch.nn.Module):
    def __init__(self, dummy_module):
        super().__init__()

        self.dummy = dummy_module

    def forward(self, batch, **kwargs):
        return self.dummy(batch), kwargs




_AGGREGATORS = {"mean": GlobalMean,
                "running_mean": RunningMean,
                "multi_running_mean": MultiRunningMean,
                "recursive": RecursiveReduction}


def get_aggregator(key):
    if key in _AGGREGATORS:
        return _AGGREGATORS[key]
    current_module = sys.modules[__name__]
    return getattr(current_module, key)
