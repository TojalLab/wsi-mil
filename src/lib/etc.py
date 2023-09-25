
import torch
import torch.utils.data
import builtins
import numpy
from PIL import Image
from scipy import linalg
from skimage.util import dtype, dtype_limits
from skimage.exposure import rescale_intensity
import tiatoolbox
import tiatoolbox.tools
import tiatoolbox.data
import torchvision

imagenet_stats = { 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225] }

def most_confused(cm, labels, top_k=5):
    import itertools
    nl = len(labels)
    d = []
    for i, j in itertools.product(range(nl), range(nl)):
        if i != j and cm[i,j] > 0:
            d.append((labels[i], labels[j], cm[i,j].item()))
    d = sorted(d, reverse=True, key=lambda t: t[2])
    return d[:top_k]

def hms(td):
    h,r = divmod(td.total_seconds(), 3600)
    m,s = divmod(r, 60)
    if h > 0:
        return '%02d:%02d:%02.f'%(h,m,s)
    else:
        return '%02d:%02.f'%(m,s)

class FromDictModule(torch.nn.Module):
    def __init__(self, key):
        super().__init__()
        self.key = key
    def forward(self, x):
        return x[self.key]

def create_imagenet_feature_extractor(torchvision_model_name, feature_layer, pooling_layer=torch.nn.AdaptiveAvgPool2d(1)):
    import torchvision
    model = getattr(torchvision.models, torchvision_model_name)(pretrained=True)
    fe = torchvision.models.feature_extraction.create_feature_extractor(model, return_nodes=[feature_layer]).eval()
    normalization_layer = torchvision.transforms.Normalize(**imagenet_stats)
    return torch.nn.Sequential(
        normalization_layer,
        fe,
        FromDictModule(key=feature_layer),
        pooling_layer,
        torch.nn.Flatten()
    ).eval()

def read_metadata(path, check_cols=['case_id','slide_id']):
    import pandas as pd
    tbl = pd.read_table(path,sep=None,engine='python')
    for cn in check_cols:
        if cn not in tbl.columns:
            raise Exception(f'{cn} column not found on table: {path}')
    return tbl

def create_progress_ctx():
    import rich.progress
    return rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(),
        rich.progress.TimeElapsedColumn(),
        speed_estimate_period=1e6
    )

class RandomRotate90(torch.nn.Module):
    def __init__(self, p=0.5, dim0=1, dim1=2):
        super().__init__()
        self.p = p
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, img):
        if torch.rand(1) < self.p:
            return torch.transpose(img, self.dim0, self.dim1)
        return img

rgb_from_hed = numpy.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]]).astype('float32')
hed_from_rgb = linalg.inv(rgb_from_hed).astype('float32')

def rgb2hed(rgb):
    return separate_stains(rgb, hed_from_rgb)

def hed2rgb(hed):
    return combine_stains(hed, rgb_from_hed)

def separate_stains(rgb, conv_matrix):
    rgb = dtype.img_as_float(rgb, force_copy=True).astype('float32')
    rgb += 2
    stains = numpy.dot(numpy.reshape(-numpy.log(rgb), (-1, 3)), conv_matrix)
    return numpy.reshape(stains, rgb.shape)

def combine_stains(stains, conv_matrix):
    stains = dtype.img_as_float(stains.astype('float64')).astype('float32')  # stains are out of range [-1, 1] so dtype.img_as_float complains if not float64
    logrgb2 = numpy.dot(-numpy.reshape(stains, (-1, 3)), conv_matrix)
    rgb2 = numpy.exp(logrgb2)
    return rescale_intensity(numpy.reshape(rgb2 - 2, stains.shape),
                             in_range=(-1, 1))

class RandomHED(torch.nn.Module):
    def __init__(self, p=1., sigma=0.01, bias=0.01, channel_axis=0):
        super().__init__()
        self.p=p
        self.sigma=sigma
        self.bias=bias
        self.channel_axis=channel_axis

    def forward(self, img):
        if torch.rand(1) < self.p:
            z = rgb2hed(img.moveaxis(self.channel_axis, -1))
            z = z * numpy.random.uniform(1-self.sigma, 1+self.sigma, 3)
            z = z + numpy.random.uniform(-self.bias, self.bias, 3)
            return torch.tensor(hed2rgb(z)).moveaxis(-1, self.channel_axis)
        return img

class ColorNorm(torch.nn.Module):
    def __init__(self, ref_image=None):
        super().__init__()
        self.normalizer = tiatoolbox.tools.stainnorm.MacenkoNormalizer()
        if ref_image is None:
            self.normalizer.fit(tiatoolbox.data.stain_norm_target())
            print(" >> Normalizer fits with DEFAULT image")
        else:
            image = Image.open(ref_image)
            image = numpy.asarray(image)[:,:,:3].copy()
            print(" >> Normalizer fits with ref_image:", image.shape)

            self.normalizer.fit(image)

        self.tt = torchvision.transforms.ToTensor()

    def forward(self, img):
        try:
            t = self.normalizer.transform(numpy.asarray(img).copy())
            return self.tt(t)
        except (ValueError, numpy.linalg.LinAlgError) as e:
            return self.tt(img) # empty image