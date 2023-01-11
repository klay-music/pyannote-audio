import numpy as np
from pyannote.audio import Inference, Pipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation
import torch
from typing import Optional


HF_URI = "pyannote/voice-activity-detection"


class VAD:
    """This detects the vocal activity signal from an audio signal using the
    approach described by pyannote.

    An instance of Inference can easily be retrieved using the helper function:

        `from pyannote.audio.pipelines.utils import get_inference`
    """

    sr = 16000

    def __init__(
        self,
        auth_token: str,
        onset: float = 0.5,
        offset: float = 0.5,
        min_duration_on: float = 0.1,
        min_duration_off: float = 0.1,
        inference: Optional[Inference] = None
    ):
        pipeline = Pipeline.from_pretrained(HF_URI, use_auth_token=auth_token)
        self.inference = pipeline._segmentation
        self.binarize = Binarize(
            onset=onset,
            offset=offset,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

    def __call__(self, signal: np.ndarray, sr: int) -> Annotation:
        assert sr == self.sr, f"Invalid sample rate: {sr}, please set to 16000"
        signal = torch.tensor(
            signal, dtype=torch.float32, device=self.inference.model.device
        )
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)

        segmentation = self.inference.slide(signal, sr)
        return self.binarize(segmentation)
