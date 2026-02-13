import librosa
import numpy as np

def extract_dna(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    signal = y[0:512]

    # Standard Librosa Pipeline
    # 1. Window
    windowed = signal * np.hanning(512)
    # 2. FFT
    spectrum = np.abs(np.fft.rfft(windowed))
    # 3. Mel
    mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=40)
    mel_energies = np.dot(mel_basis, spectrum)
    # 4. Log
    log_mel = np.log(mel_energies + 1e-6)
    # 5. DCT
    # We used 'ortho' in our export script, so we must use it here
    from scipy.fftpack import dct
    mfccs = dct(log_mel, type=2, norm='ortho')

    print(f"✅ Python Results (Manual Librosa):")
    print(mfccs[:5])

if __name__ == "__main__":
    extract_dna("test_sine.wav")