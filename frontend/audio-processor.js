/**
 * AudioCaptureProcessor
 *
 * Two-gate VAD:
 *   Gate 1 — RMS energy must exceed threshold (tunable)
 *   Gate 2 — Spectral centroid must fall in the voiced-speech band (300–3400 Hz)
 *
 * Both gates must be true for N consecutive frames before "user-started-speaking"
 * is fired.  This rejects:
 *   • Low-frequency hum / fan / A/C (fails centroid gate)
 *   • Brief transients like a single knock (fails frame-count gate)
 *   • Distant conversation / TV (fails energy gate at normal mic distances)
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // ── Energy gate ────────────────────────────────────────────────
        // Raise if background noise still triggers; lower if your mic is quiet.
        this._energyThreshold = 0.25;   // RMS to start considering speech
        this._silenceThreshold = 0.30;   // RMS below which we count silence

        // ── Temporal gate ──────────────────────────────────────────────
        // How many consecutive "voiced" frames before we declare speech.
        // At 128 samples / 16 kHz each frame ≈ 8 ms → 4 frames ≈ 32 ms
        this._speechConfirmFrames = 4;
        this._silenceConfirmFrames = 12;  // ~96 ms of quiet to end speech

        // ── Spectral gate ──────────────────────────────────────────────
        this._fftSize = 256;          // must be power-of-two
        this._sampleRate = 16000;
        this._voiceLowHz = 300;
        this._voiceHighHz = 3400;

        // ── State ──────────────────────────────────────────────────────
        this._isSpeaking = false;
        this._speechFrameCount = 0;
        this._silenceFrameCount = 0;

        // Pre-compute Hann window once
        this._hannWindow = new Float32Array(this._fftSize);
        for (let i = 0; i < this._fftSize; i++) {
            this._hannWindow[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (this._fftSize - 1)));
        }

        // Scratch buffer for FFT input (zero-padded if frame < fftSize)
        this._fftBuf = new Float32Array(this._fftSize);
    }

    // ── Goertzel-free real DFT (only the bins we care about) ────────────
    // Full radix-2 FFT would be faster but adds complexity; at 256 pts this
    // runs comfortably within the 128-sample render quantum budget.
    _spectralCentroid(frame) {
        const N = this._fftSize;
        const buf = this._fftBuf;
        const win = this._hannWindow;

        // Fill + window
        const copyLen = Math.min(frame.length, N);
        for (let i = 0; i < copyLen; i++) buf[i] = frame[i] * win[i];
        for (let i = copyLen; i < N; i++) buf[i] = 0;

        // DFT — compute only positive frequencies
        let weightedSum = 0;
        let totalPower = 0;
        const halfN = N >> 1;

        for (let k = 1; k < halfN; k++) {
            const freq = (k * this._sampleRate) / N;
            let re = 0, im = 0;
            for (let n = 0; n < N; n++) {
                const angle = (2 * Math.PI * k * n) / N;
                re += buf[n] * Math.cos(angle);
                im -= buf[n] * Math.sin(angle);
            }
            const power = re * re + im * im;
            weightedSum += freq * power;
            totalPower += power;
        }

        return totalPower < 1e-10 ? 0 : weightedSum / totalPower;
    }

    // ── Check what fraction of energy sits in the voiced band ───────────
    _voiceBandEnergyRatio(frame) {
        const N = this._fftSize;
        const buf = this._fftBuf;
        const win = this._hannWindow;

        const copyLen = Math.min(frame.length, N);
        for (let i = 0; i < copyLen; i++) buf[i] = frame[i] * win[i];
        for (let i = copyLen; i < N; i++) buf[i] = 0;

        const lowBin = Math.floor((this._voiceLowHz * N) / this._sampleRate);
        const highBin = Math.ceil((this._voiceHighHz * N) / this._sampleRate);
        const halfN = N >> 1;

        let voiceEnergy = 0;
        let totalEnergy = 0;

        for (let k = 1; k < halfN; k++) {
            let re = 0, im = 0;
            for (let n = 0; n < N; n++) {
                const angle = (2 * Math.PI * k * n) / N;
                re += buf[n] * Math.cos(angle);
                im -= buf[n] * Math.sin(angle);
            }
            const power = re * re + im * im;
            totalEnergy += power;
            if (k >= lowBin && k <= highBin) voiceEnergy += power;
        }

        return totalEnergy < 1e-10 ? 0 : voiceEnergy / totalEnergy;
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0];

        // ── Gate 1: RMS energy ─────────────────────────────────────────
        let sumSq = 0;
        for (let i = 0; i < channelData.length; i++) sumSq += channelData[i] * channelData[i];
        const rms = Math.sqrt(sumSq / channelData.length);

        let voiceDetected = false;

        if (rms > this._energyThreshold) {
            // ── Gate 2: voiced-band energy ratio ──────────────────────
            // Skip the (expensive) spectral check on truly loud frames —
            // at very high RMS it's almost certainly the close-talker.
            const ratio = rms > 0.12 ? 1.0 : this._voiceBandEnergyRatio(channelData);
            voiceDetected = ratio >= 0.45; // ≥45 % of energy in 300–3400 Hz
        }

        if (voiceDetected) {
            this._silenceFrameCount = 0;
            this._speechFrameCount++;

            if (this._speechFrameCount >= this._speechConfirmFrames && !this._isSpeaking) {
                this._isSpeaking = true;
                this.port.postMessage({ type: 'user-started-speaking' });
            }
        } else if (rms < this._silenceThreshold) {
            this._speechFrameCount = 0;
            if (this._isSpeaking) {
                this._silenceFrameCount++;
                if (this._silenceFrameCount >= this._silenceConfirmFrames) {
                    this._isSpeaking = false;
                    this._silenceFrameCount = 0;
                }
            }
        }

        // Always forward PCM to the main thread for streaming to server.
        this.port.postMessage(channelData);
        return true;
    }
}

registerProcessor('audio-capture', AudioCaptureProcessor);