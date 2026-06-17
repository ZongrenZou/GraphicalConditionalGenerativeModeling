import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from flax import serialization


def save_model(
    ckpt_dir: str | os.PathLike,
    params: Any,
    model,
    model_cfg,
    extras: Dict[str, np.ndarray] | None = None,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    b = serialization.to_bytes(params)
    (ckpt_dir / "params.msgpack").write_bytes(b)

    if is_dataclass(model_cfg):
        model_cfg = asdict(model_cfg)
    arch = {
        "hidden_dims": list(
            model_cfg.get("hidden_dims", getattr(model, "hidden_dims", []))
        ),
        "dim_out": int(getattr(model, "dim_out", model_cfg.get("dim_out"))),
    }
    (ckpt_dir / "model_cfg.json").write_text(json.dumps(arch))

    if extras:
        np.savez(
            ckpt_dir / "extras.npz", **{k: np.asarray(v) for k, v in extras.items()}
        )


def load_model(ckpt_dir: str | os.PathLike, VelocityMLP_cls):
    ckpt_dir = Path(ckpt_dir)

    params = serialization.msgpack_restore((ckpt_dir / "params.msgpack").read_bytes())

    arch = json.loads((ckpt_dir / "model_cfg.json").read_text())
    model = VelocityMLP_cls(
        hidden_dims=tuple(arch["hidden_dims"]), dim_out=int(arch["dim_out"])
    )

    extras_path = ckpt_dir / "extras.npz"
    extras = (
        dict(np.load(extras_path, allow_pickle=False)) if extras_path.exists() else {}
    )

    return params, model, extras
