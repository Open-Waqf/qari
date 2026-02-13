import numpy as np
import librosa
from scipy.fftpack import dct

def get_meyda_mfcc(signal, sr, n_mfcc=40, n_fft=512, hop_length=256):
    # 1. Windowing (Hanning) - Matches Meyda
    window = np.hanning(n_fft)

    # 2. STFT (Manual to ensure exact FFT match)
    # We only take the first frame for our test
    if len(signal) < n_fft:
        pad_width = n_fft - len(signal)
        signal = np.pad(signal, (0, pad_width), mode='constant')

    frame = signal[:n_fft] * window
    spectrum = np.abs(np.fft.rfft(frame)) # Magnitude Spectrum

    # 3. Mel Filterbank
    # Meyda uses the standard Librosa filterbank logic
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mfcc)

    # 4. Mel Energies
    mel_energies = np.dot(mel_basis, spectrum)

    # 5. Logarithm (The Critical Step)
    # Meyda adds a tiny epsilon (1e-6) to avoid log(0)
    # And it uses Natural Log (log), NOT log10 (db)
    log_mel_energies = np.log(mel_energies + 1e-6)

    # 6. DCT (Discrete Cosine Transform) - Type 2, Orthogonal
    mfccs = dct(log_mel_energies, type=2, norm='ortho')

    return mfccs