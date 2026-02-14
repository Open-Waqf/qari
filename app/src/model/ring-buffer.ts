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
     */
    clear() {
        this.index = 0;
        this.isFull = false;
        this.buffer.fill(0);
    }

    /**
     * Optimized write using block memory operations (memcpy)
     * instead of iterating element-by-element.
     */
    write(chunk: Float32Array) {
        // Safety: If chunk is larger than buffer, just take the end
        if (chunk.length >= this.size) {
            this.buffer.set(chunk.subarray(chunk.length - this.size));
            this.index = 0;
            this.isFull = true;
            return;
        }

        const freeSpace = this.size - this.index;

        if (chunk.length <= freeSpace) {
            // Case 1: Chunk fits in the remaining space (No wrap)
            this.buffer.set(chunk, this.index);
            this.index += chunk.length;

            // Boundary check: if we filled exactly to the end
            if (this.index === this.size) {
                this.index = 0;
                this.isFull = true;
            }
        } else {
            // Case 2: Chunk wraps around
            const part1 = chunk.subarray(0, freeSpace);
            const part2 = chunk.subarray(freeSpace);

            this.buffer.set(part1, this.index); // Fill to end
            this.buffer.set(part2, 0);          // Continue from start

            this.index = part2.length;
            this.isFull = true;
        }
    }

    read(): Float32Array {
        if (!this.isFull) return new Float32Array(0);

        const result = new Float32Array(this.size);

        // 1. Copy oldest data (from index to end) to start of result
        const tail = this.buffer.subarray(this.index);
        result.set(tail, 0);

        // 2. Copy newest data (from 0 to index) to end of result
        const head = this.buffer.subarray(0, this.index);
        result.set(head, tail.length);

        return result;
    }
}