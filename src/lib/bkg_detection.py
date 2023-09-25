
import torch
import numpy
import torch.utils.data
import torchvision
import openslide
import onnxruntime as ort
from lib.etc import RandomRotate90, RandomHED
from lib.etc import ColorNorm

MODEL_9TISSUES_PATH = 'pretrained_models/9tissues_0.956.onnx'
MODEL_9TISSUES_LABELS = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
MODEL_9TISSUES_LABEL_COLORS = {
    'ADI': (50,50,50), # black
    'BACK': (0,0,0),  # gray
    'DEB': (255,255,51), # yellow
    'MUC': (166,86,40), # brown
    'LYM': (255,127,0), # orange
    'MUS': (152,78,163), # purple
    'NORM': (77,175,74), # green
    'STR': (55,126,184), # blue
    'TUM': (228,26,28) # red
}

class TiledSlide:
    def __init__(self, slide_path, *, tile_size=224, step_size=224, target_mpp=0.5, orig_mpp=None, coords=None):

        self.meta = {
            'slide_path': slide_path,
            'target_mpp': target_mpp,
            'level': 0,
            'tile_size': tile_size,
            'step_size': step_size
        }
        
        self.slide = openslide.open_slide(slide_path)
        _ = self.slide.get_thumbnail((200,200)) # force load openslide
        
        MPP_DIFF_TOLERANCE = 0.02
        
        if orig_mpp is None:
            self.meta['orig_mpp'] = float(self.slide.properties['aperio.MPP'])
        else:
            self.meta['orig_mpp'] = orig_mpp
        
        if (self.meta['orig_mpp']-MPP_DIFF_TOLERANCE) > target_mpp:
            raise Exception(f"slide does not have enough resolution: {self.meta['orig_mpp']} > {target_mpp}")
        
        if abs(target_mpp - self.meta['orig_mpp']) > MPP_DIFF_TOLERANCE:
            self.meta['scale'] = target_mpp / self.meta['orig_mpp']
        else:
            self.meta['scale'] = 1
        
        if coords is not None:
            self.coords = coords
        else:
            self.gen_all_coords()

    def gen_all_coords(self):
        tile_size = round(self.meta['tile_size'] * self.meta['scale'])
        step_size = round(self.meta['step_size'] * self.meta['scale'])

        coords = []
        i = 0
        while i+step_size < self.slide.level_dimensions[self.meta['level']][0]:
            j = 0
            while j+step_size < self.slide.level_dimensions[self.meta['level']][1]:
                coords.append([i,j])
                j += step_size
            i += step_size
        self.coords = torch.tensor(coords)

    @classmethod
    def from_tiles_pt(cls, pt_path):
        dd = torch.load(pt_path)
        return cls(
            slide_path=dd['metadata']['slide_path'],
            tile_size=dd['metadata']['tile_size'],
            step_size=dd['metadata']['step_size'],
            target_mpp=dd['metadata']['target_mpp'],
            orig_mpp=dd['metadata']['orig_mpp'],
            coords=dd['coords']
        )
    
    def save_all_tiles_pt(self, out_path):
        torch.save({
            'coords': self.coords,
            'metadata': self.meta
        }, out_path)

class TileDataset(torch.utils.data.Dataset):
    def __init__(self, tiled_slide, color_norm=None):
        self.tiled_slide = tiled_slide
        self.coords = tiled_slide.coords
        ts = round(tiled_slide.meta['tile_size'] * tiled_slide.meta['scale'])
        self.size = (ts,ts)
        self.level = tiled_slide.meta['level']
        self.tfms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((tiled_slide.meta['tile_size'], tiled_slide.meta['tile_size'])),
            ColorNorm(color_norm),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomVerticalFlip(),
            # torchvision.transforms.ToTensor(),
            # RandomRotate90(),
            #RandomHED(p=1.),
            #torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ])

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.tfms(self.tiled_slide.slide.read_region(self.coords[idx].tolist(), level=self.level, size=self.size).convert('RGB'))


class BkgDetection:
    """runs inference with the 9tissues NN and saves coords for patches without background
    example:
      obj = dlutils.bkg_detection.BkgDetection(TiledSlide(slide_path))
      obj.inference()
      tn = obj.preds_thumbnail()
      # show_image(tn)
      obj.save_patches('test.h5')
    """
    
    def __init__(self, tiled_slide):
        self.tiled_slide = tiled_slide

    def inference(self):
        ds = TileDataset(self.tiled_slide)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=16, drop_last=False)

        so = ort.SessionOptions()
        so.inter_op_num_threads = 8
        so.intra_op_num_threads = 8
        sess = ort.InferenceSession(MODEL_9TISSUES_PATH, providers=['CUDAExecutionProvider'], sess_options=so)

        preds = []
        for x in dl:
            y_hat = sess.run(['9tissues'],  {'image_N_3_224_224': x.numpy()})
            preds.append(torch.tensor(y_hat[0]))
        self.preds = torch.cat(preds).cpu()

    def preds_thumbnail(self, size=(1200,1200)):
        tn = self.tiled_slide.slide.get_thumbnail(size)
        agg = torch.zeros((tn.size[1], tn.size[0], self.preds.shape[1]))
        cnt = torch.zeros((tn.size[1], tn.size[0], self.preds.shape[1]))
        
        factor = torch.tensor(tn.size) / torch.tensor(self.tiled_slide.slide.dimensions)

        ts = self.tiled_slide.meta['tile_size'] * self.tiled_slide.meta['scale']
        eps = 0.01
        for i,c in enumerate(self.tiled_slide.coords):
            
            x1, y1 = ((c * factor)+eps).round().int()
            x2, y2 = (((c+ts)*factor)+eps).round().int()
            agg[y1:y2,x1:x2] += self.preds[i]
            cnt[y1:y2,x1:x2] += 1
        agg = agg/cnt
        
        ct = torch.Tensor(list(map(lambda i: MODEL_9TISSUES_LABEL_COLORS[i], MODEL_9TISSUES_LABELS)))
        return ct[agg.argmax(dim=2)].round().int()

    def save_all_preds(self, out_path):
        wm = [ MODEL_9TISSUES_LABELS[i] for i in self.preds.max(1).indices ]
        torch.save({
            'coords': self.tiled_slide.coords,
            'metadata': self.tiled_slide.meta,
            'top_preds': wm,
        }, out_path)

    def filter_coords(self, labels_to_exclude=['ADI','BACK']):
        wm = [ MODEL_9TISSUES_LABELS[i] for i in self.preds.max(1).indices ]
        filtered_coords = []
        for i, c in enumerate(self.tiled_slide.coords):
            if wm[i] not in labels_to_exclude:
                filtered_coords.append(c)
        if len(filtered_coords) > 0:
            self.filtered_coords = torch.stack(filtered_coords)
        else:
            self.filtered_coords = torch.tensor([])

    def save_filtered_tiles_pt(self, out_path):
        if self.filtered_coords is None:
            self.filter_coords()
        torch.save({
            'coords': self.filtered_coords,
            'metadata': self.tiled_slide.meta
        }, out_path)
