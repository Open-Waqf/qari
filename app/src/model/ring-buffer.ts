export class RingBuffer {
    private buffer: Float32Array;
    private size: number;
    private index: number = 0;
    public isFull: boolean = false;

    constructor(durationSeconds: number, sampleRate: number) {
        this.size = durationSeconds * sampleRate; // e.g., 3 * 16000 = 48000
        this.buffer = new Float32Array(this.size);
    }

    /**
     * Resets the buffer state.
     * Essential for preventing "ghosting" when a user resumes reciting after a pause.
     */
    clear() {
        this.index = 0;
        this.isFull = false;
        this.buffer.fill(0); // Zero out the actual data for a clean slate
    }

    // Add new incoming chunks (e.g., the 42 samples)
    write(chunk: Float32Array) {
        for (let i = 0; i < chunk.length; i++) {
            this.buffer[this.index] = chunk[i];
            this.index++;

            // Wrap around if we hit the end
            if (this.index >= this.size) {
                this.index = 0;
                this.isFull = true;
            }
        }
    }

    // Get the last 3 seconds linearly (unwrapped)
    read(): Float32Array {
        if (!this.isFull) return new Float32Array(0);

        const result = new Float32Array(this.size);

        // Part 1: From current index to end (Oldest data)
        const part1 = this.buffer.subarray(this.index);
        // Part 2: From 0 to current index (Newest data)
        const part2 = this.buffer.subarray(0, this.index);

        result.set(part1);
        result.set(part2, part1.length);

        return result;
    }
}