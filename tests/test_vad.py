import numpy as np
import pytest

from pyannote.audio.vad import VAD


class TestVAD:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.vad = VAD()

    def test_vad__call_a(self):
        signal = np.random.rand(self.vad.sr)
        got_annotation = self.vad(signal, sr=16000)
        segments = [i[0] for i in got_annotation.itertracks()]
        assert len(segments) == 1
        assert segments[0].start > 0.0
        assert segments[0].end < 1.0
        assert segments[0].start < 0.05
        assert segments[0].end > 0.95
