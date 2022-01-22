import torch
import torchaudio
# import s3prl.hub as hub
from sklearn.preprocessing import StandardScaler
import numpy as np

class StandardScaler3D(StandardScaler):
    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)

SCALER = StandardScaler3D()

class AudioProcessor(object):
    def __init__(self, feature, num_mels, num_mfcc, log_mels, mel_fmin, mel_fmax, normalize, sample_rate, n_fft, num_freq, hop_length, win_length):
        self.feature = feature
        self.num_mels = num_mels
        self.num_mfcc = num_mfcc
        self.log_mels = log_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.normalize = normalize
        self.num_freq = num_freq
        self.hop_length = hop_length
        self.win_length = win_length

        valid_features = ['spectrogram', 'melspectrogram', 'mfcc']
        if self.feature not in valid_features:
            raise ValueError("Invalid Feature: "+str(self.feature))

    def wav2feature(self, y):
        if self.feature == 'spectrogram':
            audio_class = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        elif self.feature == 'melspectrogram':
            audio_class = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_mels=self.num_mels, f_min=self.mel_fmin, f_max=self.mel_fmax)
        elif self.feature == 'mfcc':
            audio_class = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=self.num_mfcc, log_mels=self.log_mels, melkwargs={'n_fft':self.n_fft, 'win_length':self.win_length, 'hop_length':self.hop_length, 'n_mels':self.num_mels})
        # elif self.feature == 'wav2vec2':
        #     audio_class = getattr(hub, 'wav2vec2')()
        
        feature = SCALER.fit_transform(audio_class(y))
        # feature = audio_class(y)
        feature = torch.from_numpy(feature).float()
        # print(f"FEATURE SHAPE: {feature.shape})")
        return feature

    def get_feature_from_audio_path(self, audio_path):
        return self.wav2feature(self.load_wav(audio_path))

    def get_feature_from_audio(self, wav):
        return self.wav2feature(wav)

    def load_wav(self, path):
        # load audio path and normalize to [-1, 1]
        wav, sample_rate = torchaudio.load(path, normalize=self.normalize)
        # resample audio for specific samplerate
        if sample_rate != self.sample_rate:
            resample = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            wav = resample(wav)
        return wav
