import {audioManager} from '../core/audio-manager';
import {customExtractor} from '../features/custom-extractor';

export async function checkAudioParity(audioUrl: string) {
    const context = audioManager.context;
    await customExtractor.loadConfig();

    const response = await fetch(audioUrl);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await context.decodeAudioData(arrayBuffer);

    // We need enough data for at least one frame now
    const signal = audioBuffer.getChannelData(0).slice(0, 16000); // 1 second

    // Use the new method
    const tfResult = customExtractor.extractFullClip(signal);
    const values = await tfResult.data();

    console.log("🚀 CUSTOM EXTRACTOR RUN (Full Clip Mode):");
    console.log("Output Shape:", tfResult.shape);
    console.log("First 5 Values:", values.slice(0, 5));
    tfResult.dispose();
}