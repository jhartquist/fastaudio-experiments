from utils import *
assert torch.cuda.is_available()

run_config = dict(
    # spectrum
    sample_rate=44100,
    n_fft=4096,
    n_mels=224,
    hop_length=441,
    win_length=1764,
    f_max=20000,

    # model
    arch='resnet18',

    # training
    learning_rate=1e-2,
    n_epochs=20,
    batch_size=64,
    mix_up=0.4,
    normalize=True,
    
    # data
    trial_num=1,
    fold=1,
)

run = wandb.init(
    config=run_config,
    save_code=True)

config = wandb.config

print("Config:", json.dumps(config.as_dict(), indent=2))

audio_config = AudioConfig.BasicMelSpectrogram(
    sample_rate=config.sample_rate,
    hop_length=config.hop_length,
    win_length=config.win_length,
    n_fft=config.n_fft,
    n_mels=config.n_mels,
    normalized=True,
    f_max=config.f_max)

to_spectrum = AudioToSpec.from_cfg(audio_config)
batch_tfms = [to_spectrum]
data = get_data(batch_tfms=batch_tfms, 
                sample_rate=config.sample_rate,
                batch_size=config.batch_size,
                fold=config.fold,
                seed=config.trial_num)

if config.normalize:
    stats = StatsRecorder()
    with torch.no_grad():
        for x,y in iter(data.train):
            stats.update(x)
    data.after_batch.add(SpecNormalize(stats.mean, stats.std))

    
arch = eval(config.arch)
learn = get_learner(data, arch, 
                    # if computing manual norm stats, do not use default pretrained stats
                    normalize=(not config.normalize))

cbs = []
if config.mix_up: 
    cbs.append(MixUp(config.mix_up))
cbs.append(WandbCallback(log_model=False, log_preds=False))

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate, 
                cbs=cbs)

wandb.finish()
