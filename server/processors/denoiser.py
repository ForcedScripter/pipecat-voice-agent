import torch
import torchaudio.functional as F


def _denoise(self, audio_bytes: bytes) -> bytes:
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_np /= 32768.0
        audio_np = audio_np[np.newaxis, :]  # (1, samples)
        audio_tensor = torch.from_numpy(audio_np)

        model_sr = self._df_state.sr()  # usually 48000

        # Upsample to model rate before enhancing
        if self._sample_rate != model_sr:
            audio_tensor = F.resample(audio_tensor, self._sample_rate, model_sr)

        enhanced = enhance(self._model, self._df_state, audio_tensor)

        # Downsample back to pipeline rate
        if self._sample_rate != model_sr:
            enhanced = F.resample(enhanced, model_sr, self._sample_rate)

        enhanced_np = enhanced.squeeze(0).numpy()
        enhanced_np = np.clip(enhanced_np * 32768.0, -32768, 32767).astype(np.int16)
        return enhanced_np.tobytes()

    except Exception as e:
        logger.warning("DeepFilterNet denoise failed, passing through: {}", e)
        return audio_bytes
