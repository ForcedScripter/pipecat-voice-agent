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
        /** @type {AudioBufferSourceNode | null} */
        this._currentPlaybackSource = null;
        this._audioFramesSent = 0;
        this._micHealthTimer = null;
        this._botIsSpeaking = false;
        this._flushGeneration = 0;
        // When false, drop incoming bot PCM (stale audio after an interrupt).
        this._acceptBotAudio = false;
        /** @type {ReturnType<typeof setTimeout> | null} */
        this._interruptDebounceTimer = null;

        // ── Barge-in tuning ────────────────────────────────────────────
        // How long (ms) after the local VAD fires before we act.
        // Gives time for a spurious noise to "un-trigger" before we flush.
        this._interruptDebounceMs = 220;

        // Minimum ms the bot must have been speaking before a local-VAD
        // barge-in is honoured.  Prevents a noise burst at utterance-start
        // from immediately killing the response.
        this._minBotSpeakingMsBeforeInterrupt = 400;
        this._botStartedSpeakingAt = 0;

        // Set when server VAD already triggered barge-in — skip redundant JSON interrupt.
        this._serverHandledBargeIn = false;
        /** @type {ReturnType<typeof setTimeout> | null} */
        this._serverBargeInResetTimer = null;
    }

    /* ------------------------------------------------------------------ */
    /* Public API                                                           */
    /* ------------------------------------------------------------------ */

    async connect(lang = 'hi-IN', voice = 'netra') {
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

            const params = new URLSearchParams({ lang, voice });
            const url = `${this.wsBaseUrl}?${params.toString()}`;
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

    /* ------------------------------------------------------------------ */
    /* Audio capture (mic -> server)                                        */
    /* ------------------------------------------------------------------ */

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

            const data = event.data;

            if (data && typeof data === 'object' && data.type === 'user-started-speaking') {
                // Only trigger barge-in if bot audio is actually playing
                const botIsActive = this._botIsSpeaking || this._isPlaying || this._playQueue.length > 0;
                if (botIsActive) {
                    this._scheduleLocalBargeIn();
                }
                return;
            }

            const float32 = data;
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
    }

    /* ------------------------------------------------------------------ */
    /* Incoming messages                                                    */
    /* ------------------------------------------------------------------ */

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
                    this._botIsSpeaking = true;
                    this._acceptBotAudio = true;
                    this._botStartedSpeakingAt = Date.now();
                    console.log('[VoiceAgent] bot started speaking');
                    this._onBotStartedSpeaking();
                    break;

                case 'bot-stopped-speaking':
                    this._botIsSpeaking = false;
                    this._onBotStoppedSpeaking();
                    break;

                case 'user-started-speaking':
                    // Server VAD fired — only honour it if bot is actually active.
                    // (Server may fire this on background noise too.)
                    if (this._botIsSpeaking || this._isPlaying || this._playQueue.length > 0) {
                        console.log('[VoiceAgent] server VAD: user started speaking — barge-in (playback only)');
                        this._handleBargeInPlayback(false);
                    } else {
                        console.log('[VoiceAgent] server VAD: user-started-speaking but bot is idle — ignoring');
                    }
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

    /* ------------------------------------------------------------------ */
    /* Audio playback (server -> speaker)                                   */
    /* ------------------------------------------------------------------ */

    _stopCurrentPlaybackSource() {
        const src = this._currentPlaybackSource;
        if (!src) return;
        try {
            src.onended = null;
            src.stop(0);
        } catch (_e) {
            // Already stopped or context closed — safe to ignore.
        }
        this._currentPlaybackSource = null;
    }

    _flushPlaybackQueue() {
        this._flushGeneration++;

        const hadContent = this._playQueue.length > 0 || this._isPlaying || this._currentPlaybackSource;
        this._stopCurrentPlaybackSource();
        this._playQueue = [];
        this._isPlaying = false;

        if (hadContent) {
            console.log('[VoiceAgent] flushed playback (generation now', this._flushGeneration, ')');
        }
    }

    _clearInterruptDebounce() {
        if (this._interruptDebounceTimer) {
            clearTimeout(this._interruptDebounceTimer);
            this._interruptDebounceTimer = null;
        }
    }

    _clearServerBargeInReset() {
        if (this._serverBargeInResetTimer) {
            clearTimeout(this._serverBargeInResetTimer);
            this._serverBargeInResetTimer = null;
        }
    }

    _scheduleLocalBargeIn() {
        this._clearInterruptDebounce();
        this._interruptDebounceTimer = setTimeout(() => {
            this._interruptDebounceTimer = null;
            if (!this.isConnected) return;

            const botActive = this._botIsSpeaking || this._isPlaying || this._playQueue.length > 0;
            if (!botActive) return;

            // Don't interrupt if the bot just started speaking — avoids noise
            // at the very start of a response killing it immediately.
            const msSinceBotStarted = Date.now() - this._botStartedSpeakingAt;
            if (msSinceBotStarted < this._minBotSpeakingMsBeforeInterrupt) {
                console.log('[VoiceAgent] local VAD: barge-in suppressed (bot only speaking for', msSinceBotStarted, 'ms)');
                return;
            }

            console.log('[VoiceAgent] local VAD: barge-in — stopping bot playback');
            this._handleBargeInPlayback(true);
        }, this._interruptDebounceMs);
    }

    /**
     * Stop bot audio locally. Optionally notify server (local VAD only).
     * @param {boolean} sendServerInterrupt
     */
    _handleBargeInPlayback(sendServerInterrupt) {
        const wasBotActive = this._botIsSpeaking || this._isPlaying || this._playQueue.length > 0;
        this._flushPlaybackQueue();
        this._botIsSpeaking = false;
        this._acceptBotAudio = false;

        if (!sendServerInterrupt) {
            this._serverHandledBargeIn = true;
            this._clearServerBargeInReset();
            this._serverBargeInResetTimer = setTimeout(() => {
                this._serverHandledBargeIn = false;
                this._serverBargeInResetTimer = null;
            }, 800);
            return;
        }

        if (
            wasBotActive &&
            !this._serverHandledBargeIn &&
            this.ws &&
            this.ws.readyState === WebSocket.OPEN
        ) {
            console.log('[VoiceAgent] sending interrupt to server (local-vad)');
            this.ws.send(JSON.stringify({ type: 'interrupt' }));
        }
    }

    _enqueueAudio(arrayBuffer) {
        if (!this._acceptBotAudio) return;
        this._playQueue.push(arrayBuffer);
        if (!this._isPlaying) this._playNext();
    }

    async _playNext() {
        const myGeneration = this._flushGeneration;

        if (this._playQueue.length === 0) {
            this._isPlaying = false;
            return;
        }
        this._isPlaying = true;
        const buf = this._playQueue.shift();

        try {
            if (!this.audioContext) return;
            if (this._flushGeneration !== myGeneration) return;

            const pcm16 = new Int16Array(buf);
            const float32 = new Float32Array(pcm16.length);
            for (let i = 0; i < pcm16.length; i++) {
                float32[i] = pcm16[i] / 32768;
            }
            const audioBuffer = this.audioContext.createBuffer(1, float32.length, 16000);
            audioBuffer.getChannelData(0).set(float32);

            if (this._flushGeneration !== myGeneration) return;

            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            this._currentPlaybackSource = source;

            source.onended = () => {
                if (this._currentPlaybackSource === source) {
                    this._currentPlaybackSource = null;
                }
                if (this._flushGeneration === myGeneration) {
                    this._playNext();
                }
            };

            source.start();
        } catch (err) {
            console.error('[VoiceAgent] playback error:', err);
            if (this._flushGeneration === myGeneration) {
                this._playNext();
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Helpers                                                              */
    /* ------------------------------------------------------------------ */

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
        this._botIsSpeaking = false;
        this._acceptBotAudio = false;
        this._clearInterruptDebounce();
        this._clearServerBargeInReset();
        this._serverHandledBargeIn = false;
        this._flushGeneration++;
        this._stopCurrentPlaybackSource();
        this._playQueue = [];
        this._isPlaying = false;
        this._audioFramesSent = 0;
        this._botStartedSpeakingAt = 0;
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

    /* ------------------------------------------------------------------ */
    /* Lifecycle hooks (override in subclass)                               */
    /* ------------------------------------------------------------------ */
    _onConnected() { }
    _onDisconnected() { }
    _onBotStartedSpeaking() { }
    _onBotStoppedSpeaking() { }
    _onTranscription(_text) { }
    _onThinking() { }
    _onError(_data) { }
}