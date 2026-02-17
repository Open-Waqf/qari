import numpy as np
import scipy.io.wavfile as wav

# Settings
SR = 22050
DURATION = 2.0
FREQ = 440.0 # A4 Note

# Generate pure sine wave
t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
y = 0.5 * np.sin(2 * np.pi * FREQ * t) # Amplitude 0.5

# Save as WAV (Uncompressed, perfect quality)
wav.write("test_sine.wav", SR, (y * 32767).astype(np.int16))
print("✅ Created test_sine.wav")