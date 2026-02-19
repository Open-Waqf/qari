export class RingBuffer {
    private buffer: Float32Array;
    private readBuffer: Float32Array;

    private size: number;
    private index: number = 0;
    public isFull: boolean = false;

    constructor(durationSeconds: number, sampleRate: number) {
        this.size = durationSeconds * sampleRate;
        this.buffer = new Float32Array(this.size);
        this.readBuffer = new Float32Array(this.size);
    }

    clear() {
        this.index = 0;
        this.isFull = false;
        this.buffer.fill(0);
    }

    write(chunk: Float32Array) {
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
        if (!this.isFull) return new Float32Array(0); // Rare edge case

        const tail = this.buffer.subarray(this.index);
        this.readBuffer.set(tail, 0);

        const head = this.buffer.subarray(0, this.index);
        this.readBuffer.set(head, tail.length);

        return this.readBuffer;
    }
}