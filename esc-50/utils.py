from fastai.vision.all import *
from fastaudio.core.all import *

import wandb
from fastai.callback.wandb import *


path = untar_data(URLs.ESC50)

def get_data(sample_rate=16000, 
             item_tfms=None, 
             batch_tfms=None, 
             fold=1,
             batch_size=32,
             path=path,
             seed=1):
    set_seed(seed, True)
    df = pd.read_csv(path/'meta'/'esc50.csv')
    splitter = IndexSplitter(df[df.fold == fold].index)
    audio_block = AudioBlock(sample_rate=sample_rate)
    data_block = DataBlock(
        blocks=(audio_block, CategoryBlock),
        get_x=ColReader('filename', pref=path/'audio'),
        get_y=ColReader('category'),
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms)
    data = data_block.dataloaders(df, bs=batch_size)
    return data

def get_learner(data, arch, n_channels=1, pretrained=True, normalize=True):
    return cnn_learner(data, arch,
                       config=cnn_config(n_in=n_channels),
                       pretrained=pretrained,
                       normalize=normalize,
                       loss_fn=CrossEntropyLossFlat, 
                       metrics=accuracy).to_fp16()


# courtesy of Chris Kroenke @clck10
# https://enzokro.dev/spectrogram_normalizations/2020/09/10/Normalizing-spectrograms-for-deep-learning.html
class SpecNormalize(Normalize):
    "Normalize/denorm batch of `TensorImage`"
    def encodes(self, x:TensorImageBase): return (x-self.mean) / self.std
    def decodes(self, x:TensorImageBase):
        f = to_cpu if x.device.type=='cpu' else noop
        return (x*f(self.std) + f(self.mean))
    
class StatsRecorder:
    def __init__(self, red_dims=(0,2,3)):
        """Accumulates normalization statistics across mini-batches.
        ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims,keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            if data.shape[1] != self.ndimensions:
                raise ValueError('Data dims do not match previous observations.')
            
            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd  = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = torch.sqrt(self.std)
                                 
            # update total number of seen samples
            self.nobservations += n
