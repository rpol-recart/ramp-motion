from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import BaseModel, Field, ConfigDict


class ROI(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_id: str
    rtsp_url: str
    roi: ROI


class MotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thresh: float = Field(gt=0.0, lt=1.0)
    t_quiet_sec: float = Field(gt=0.0)
    t_enter_sec: float = Field(gt=0.0)
    min_blob_area_px: int = Field(ge=0)
    calib_rate_limit_sec: float = Field(ge=0.0)
    t_stale_sec: float = Field(gt=0.0)


class Mog2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: int = Field(gt=0)
    var_threshold: float = Field(gt=0.0)
    detect_shadows: bool


class ClaheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_limit: float = Field(gt=0.0)
    tile_grid: Tuple[int, int]


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_fps: int = Field(gt=0)
    rtsp_reconnect_interval_sec: int = Field(ge=0)
    rtsp_reconnect_attempts: int   # -1 means infinite
    batch_size_auto: bool


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: str
    jpeg_quality: int = Field(ge=1, le=100)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: str = "INFO"
    events_log: str


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    streams: List[StreamConfig]
    motion: MotionConfig
    mog2: Mog2Config
    clahe: ClaheConfig
    pipeline: PipelineConfig
    storage: StorageConfig
    logging: LoggingConfig


def load_config(path: Path) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
