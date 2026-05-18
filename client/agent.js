/**
 * VoiceAgent -- handles WebSocket <-> mic/speaker streaming.
 *
 * Lifecycle hooks (override in subclass):
 *   _onConnected()
 *   _onDisconnected()
 *   _onBotStartedSpeaking()
 *   _onBotStoppedSpeaking()
 *   _onTranscription(text)
 *   _onThinking()
 *   _onError(data)
 */
class VoiceAgent {
    constructor(wsBaseUrl) {
        this.wsBaseUrl = wsBaseUrl;
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.workletNode = null;
        this.isConnected = false;
        this._playQueue = [];
        this._isPlaying = false;
        this._playbackTime = 0;
        this._audioFramesSent = 0;
        this._micHealthTimer = null;
    }

    /* Public API */

    async connect(lang = 'hi-IN', voice = 'shubh') {
        console.log('[VoiceAgent] connect() called, lang:', lang, 'voice:', voice);
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });
            console.log('[VoiceAgent] mic access granted');

            const url = `${this.wsBaseUrl}?lang=${lang}&voice=${voice}`;
            console.log('[VoiceAgent] connecting to WebSocket:', url);
            this.ws = new WebSocket(url);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = async () => {
                console.log('[VoiceAgent] WebSocket connected');
                this.isConnected = true;
                await this._startAudioCapture();
                this._onConnected();
                this._startMicHealthCheck();
            };

            this.ws.onmessage = (event) => this._handleMessage(event);

            this.ws.onclose = (event) => {
                console.log('[VoiceAgent] WebSocket closed, code:', event.code, 'reason:', event.reason);
                this._cleanup();
                this._onDisconnected();
            };

            this.ws.onerror = (err) => {
                console.error('[VoiceAgent] WS error', err);
                this._onError({ message: 'WebSocket error' });
            };
        } catch (err) {
            console.error('[VoiceAgent] connect failed', err);
            this._onError({ message: err.message });
        }
    }

    disconnect() {
        console.log('[VoiceAgent] disconnect() called');
        if (this.ws) this.ws.close();
        this._cleanup();
        this._onDisconnected();
    }

    /* Audio capture (mic -> server) */

    async _startAudioCapture() {
        console.log('[VoiceAgent] starting audio capture...');
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        console.log('[VoiceAgent] AudioContext state:', this.audioContext.state, '| sampleRate:', this.audioContext.sampleRate);

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        await this.audioContext.audioWorklet.addModule('audio-processor.js');
        console.log('[VoiceAgent] AudioWorklet loaded');

        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-capture');

        this.workletNode.port.onmessage = (event) => {
            if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
            const float32 = event.data;
            const pcm16 = this._float32ToPCM16(float32);
            this.ws.send(pcm16);
            this._audioFramesSent += 1;
            if (this._audioFramesSent === 1) {
                console.log('[VoiceAgent] first audio frame sent, samples:', float32.length);
            }
            if (this._audioFramesSent % 100 === 0) {
                console.log('[VoiceAgent] audio frames sent so far:', this._audioFramesSent);
            }
        };

        source.connect(this.workletNode);
        console.log('[VoiceAgent] audio capture pipeline connected');
        this._drainPlayQueue();
    }

    /* Incoming messages */

    _handleMessage(event) {
        if (event.data instanceof ArrayBuffer) {
            console.log('[VoiceAgent] binary audio received, bytes:', event.data.byteLength);
            this._enqueueAudio(event.data);
            return;
        }

        console.log('[VoiceAgent] text message from server:', event.data);
        try {
            const msg = JSON.parse(event.data);
            const type = msg.type || '';
            const data = msg.data || {};
            console.log('[VoiceAgent] RTVI type:', type);

            switch (type) {
                case 'bot-ready':
                    console.log('[VoiceAgent] bot ready, version:', data.version);
                    break;
                case 'bot-started-speaking':
                    this._onBotStartedSpeaking();
                    break;
                case 'bot-stopped-speaking':
                    this._onBotStoppedSpeaking();
                    break;
                case 'user-transcription': {
                    const text = data.text || msg.text || '';
                    console.log('[VoiceAgent] transcription:', text);
                    if (text) this._onTranscription(text);
                    break;
                }
                case 'bot-llm-started':
                    this._onThinking();
                    break;
                case 'error':
                    console.error('[VoiceAgent] error from server:', msg);
                    this._onError(data);
                    break;
                case 'error-response':
                    console.error('[VoiceAgent] error-response:', data);
                    break;
                default:
                    console.log('[VoiceAgent] unhandled type:', type, msg);
            }
        } catch (e) {
            console.warn('[VoiceAgent] failed to parse JSON:', e.message);
        }
    }

    /* Audio playback (server -> speaker) */

    _enqueueAudio(arrayBuffer) {
        this._playQueue.push(arrayBuffer);
        this._drainPlayQueue();
    }

    /**
     * Schedule TTS chunks on the Web Audio timeline so chunks play back-to-back
     * instead of waiting for each buffer to finish (which added seconds of delay).
     */
    _drainPlayQueue() {
        if (!this.audioContext) return;

        const sampleRate = this.audioContext.sampleRate || 16000;

        while (this._playQueue.length > 0) {
            const buf = this._playQueue.shift();
            try {
                const pcm16 = new Int16Array(buf);
                const float32 = new Float32Array(pcm16.length);
                for (let i = 0; i < pcm16.length; i++) {
                    float32[i] = pcm16[i] / 32768;
                }
                const audioBuffer = this.audioContext.createBuffer(1, float32.length, sampleRate);
                audioBuffer.getChannelData(0).set(float32);

                const source = this.audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(this.audioContext.destination);

                const now = this.audioContext.currentTime;
                if (this._playbackTime < now) {
                    this._playbackTime = now;
                }
                source.start(this._playbackTime);
                this._playbackTime += audioBuffer.duration;
            } catch (err) {
                console.error('[VoiceAgent] playback error:', err);
            }
        }
    }

    /* Helpers */

    _float32ToPCM16(float32) {
        const buffer = new ArrayBuffer(float32.length * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < float32.length; i++) {
            const s = Math.max(-1, Math.min(1, float32[i]));
            view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
        return buffer;
    }

    _cleanup() {
        this.isConnected = false;
        this._playQueue = [];
        this._isPlaying = false;
        this._playbackTime = 0;
        this._audioFramesSent = 0;
        if (this._micHealthTimer) { clearTimeout(this._micHealthTimer); this._micHealthTimer = null; }
        if (this.workletNode) { this.workletNode.disconnect(); this.workletNode = null; }
        if (this.audioContext) { this.audioContext.close().catch(() => { }); this.audioContext = null; }
        if (this.mediaStream) { this.mediaStream.getTracks().forEach(t => t.stop()); this.mediaStream = null; }
        console.log('[VoiceAgent] cleanup complete');
    }

    _startMicHealthCheck() {
        if (this._micHealthTimer) clearTimeout(this._micHealthTimer);
        this._audioFramesSent = 0;
        this._micHealthTimer = setTimeout(() => {
            if (!this.isConnected) return;
            if (this._audioFramesSent === 0) {
                console.error('[VoiceAgent] mic health check FAILED');
                this._onError({ message: 'No microphone audio detected.' });
            } else {
                console.log('[VoiceAgent] mic health check OK, frames sent:', this._audioFramesSent);
            }
        }, 5000);
    }

    /* Lifecycle hooks */
    _onConnected() { }
    _onDisconnected() { }
    _onBotStartedSpeaking() { }
    _onBotStoppedSpeaking() { }
    _onTranscription(_text) { }
    _onThinking() { }
    _onError(_data) { }
}