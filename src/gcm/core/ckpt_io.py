import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
from flax import serialization


def save_model(
    ckpt_dir: str | os.PathLike,
    params: Any,
    model,
    model_cfg: Mapping[str, Any] | Any,
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    b = serialization.to_bytes(params)
    (ckpt_dir / "params.msgpack").write_bytes(b)

    if is_dataclass(model_cfg):
        cfg_dict = asdict(cast(Any, model_cfg))
    elif isinstance(model_cfg, Mapping):
        cfg_dict = dict(model_cfg)
    else:
        raise TypeError(f"Unsupported model_cfg type: {type(model_cfg)!r}")

    dim_out = getattr(model, "dim_out", cfg_dict.get("dim_out"))
    if dim_out is None:
        raise ValueError("Could not determine model dim_out for checkpoint metadata")

    arch = {
        "hidden_dims": list(
            cfg_dict.get("hidden_dims", getattr(model, "hidden_dims", []))
        ),
        "dim_out": int(dim_out),
    }
    (ckpt_dir / "model_cfg.json").write_text(json.dumps(arch))

    if extras:
        arrays = {k: np.asarray(v) for k, v in extras.items()}
        np.savez(ckpt_dir / "extras.npz", allow_pickle=False, **arrays)


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
