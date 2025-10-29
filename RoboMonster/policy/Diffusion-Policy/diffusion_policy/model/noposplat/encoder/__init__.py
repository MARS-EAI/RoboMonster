from typing import Optional

from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg, EncoderNoPoSplat

ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
}

EncoderCfg = EncoderNoPoSplatCfg


def get_encoder(cfg: EncoderCfg) -> Encoder:
    encoder_cls, encoder_visualize = ENCODERS[cfg.name]
    encoder = encoder_cls(cfg)
    return encoder
