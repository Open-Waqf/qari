import {audioManager} from '../core/audio-manager';
import {customExtractor, downsampleBuffer} from '../features/custom-extractor';

//run in console await window.checkParity("/test_sine.wav");
export async function checkAudioParity(audioUrl: string) {
    const context = audioManager.context;
    await customExtractor.loadConfig();

    console.log(`🎛️ System Sample Rate: ${context.sampleRate} Hz`);

    // 2. Load & Decode
    const response = await fetch(audioUrl);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await context.decodeAudioData(arrayBuffer);

    // 3. Get Raw Data
    let signal = audioBuffer.getChannelData(0);

    // 🛑 CRITICAL FIX: Downsample to 16000 Hz
    if (audioBuffer.sampleRate !== 16000) {
        console.warn(`⚠️ Downsampling from ${audioBuffer.sampleRate} to 16000 Hz...`);
        signal = downsampleBuffer(signal, audioBuffer.sampleRate, 16000) as any;
    }

    // 4. Slice exactly 1 frame (512 samples) for direct comparison
    // Python took frame[0:512], so we do the same.
    const frame = signal.slice(0, 512);

    // 5. Run Extractor
    // We pass just this small frame to see the exact numbers
    // Note: You might need to adjust 'extractFullClip' to 'extract' if you want frame-by-frame
    // But for now let's just feed the frame.
    const tfResult = customExtractor.extractFullClip(frame);
    const values = await tfResult.data();

    console.log("🚀 PARITY RESULT (First 5 values):");
    console.log("Python Truth: [-52.72, 34.56, -10.15, 9.50, -10.44]");
    console.log("App Output:  ", values.slice(0, 5));

    tfResult.dispose();
    context.close();
}