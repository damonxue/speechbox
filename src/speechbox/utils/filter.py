import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch


def collect_chunks(
    audio: np.ndarray | torch.Tensor, chunks: List[dict], sampling_rate: int = 16000
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """Collects audio chunks."""
    if not chunks:
        chunk_metadata = {
            "start_time": 0,
            "end_time": 0,
        }
        return [np.array([], dtype=np.float32)], [chunk_metadata]

    audio_chunks = []
    chunks_metadata = []
    for chunk in chunks:
        chunk_metadata = {
            "start_time": chunk["start"] / sampling_rate,
            "end_time": chunk["end"] / sampling_rate,
        }
        audio_chunks.append(audio[chunk["start"] : chunk["end"]])
        chunks_metadata.append(chunk_metadata)
    return audio_chunks, chunks_metadata


def get_assets_path():
    """Returns the path to the assets directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")



class SileroVADModel:
    def __init__(self, encoder_path, decoder_path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4

        self.encoder_session = onnxruntime.InferenceSession(
            encoder_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ):
        assert (
            audio.ndim == 2
        ), "Input should be a 2D array with size (batch_size, num_samples)"
        assert (
            audio.shape[1] % num_samples == 0
        ), "Input size should be a multiple of num_samples"

        batch_size = audio.shape[0]

        state = np.zeros((2, batch_size, 128), dtype="float32")
        context = np.zeros(
            (batch_size, context_size_samples),
            dtype="float32",
        )

        batched_audio = audio.reshape(batch_size, -1, num_samples)
        context = batched_audio[..., -context_size_samples:]
        context[:, -1] = 0
        context = np.roll(context, 1, 1)
        batched_audio = np.concatenate([context, batched_audio], 2)

        batched_audio = batched_audio.reshape(-1, num_samples + context_size_samples)

        encoder_batch_size = 10000
        num_segments = batched_audio.shape[0]
        encoder_outputs = []
        for i in range(0, num_segments, encoder_batch_size):
            encoder_output = self.encoder_session.run(
                None, {"input": batched_audio[i : i + encoder_batch_size]}
            )[0]
            encoder_outputs.append(encoder_output)

        encoder_output = np.concatenate(encoder_outputs, axis=0)
        encoder_output = encoder_output.reshape(batch_size, -1, 128)

        decoder_outputs = []
        for window in np.split(encoder_output, encoder_output.shape[1], axis=1):
            out, state = self.decoder_session.run(
                None, {"input": window.squeeze(1), "state": state}
            )
            decoder_outputs.append(out)

        out = np.stack(decoder_outputs, axis=1).squeeze(-1)
        return out


def get_vad_model():
    """Returns the VAD model instance."""
    encoder_path = os.path.join(get_assets_path(), "silero_encoder_v5.onnx")
    decoder_path = os.path.join(get_assets_path(), "silero_decoder_v5.onnx")
    return SileroVADModel(encoder_path, decoder_path)