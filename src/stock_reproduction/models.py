import importlib.util, sys
from pathlib import Path
import torch

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(module); return module

def load_master(home: Path, config: dict):
    upstream = home / "upstream"; sys.path.insert(0, str(upstream))
    try: module = _load("master_upstream_model", upstream / "master.py")
    finally: sys.path.pop(0)
    c = config["model"]
    return module.MASTER(d_feat=158, d_model=c["hidden_size"], t_nhead=c["temporal_heads"], s_nhead=c["spatial_heads"], T_dropout_rate=c["dropout"], S_dropout_rate=c["dropout"], gate_input_start_index=158, gate_input_end_index=221, beta=c["beta"])

def load_sspt(home: Path, config: dict, input_size: int = 25):
    module = _load("sspt_upstream_model", home / "upstream/model.py"); c = config["model"]
    return module.TransformerStockPrediction(input_size=input_size, num_class=1, hidden_size=c["hidden_size"], num_feat_att_layers=c["feature_attention_layers"], num_pre_att_layers=c["prediction_attention_layers"], num_heads=c["heads"], days=90, dropout=c["dropout"])

def corrected_pairwise_rank_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = prediction.reshape(-1), target.reshape(-1)
    return torch.relu(-(prediction[:, None]-prediction[None, :])*(target[:, None]-target[None, :])).mean()

def upstream_sspt_rank_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = prediction.reshape(-1, 1), target.reshape(-1, 1); ones = torch.ones_like(prediction)
    return torch.relu((prediction@ones.T-ones@prediction.T)*(ones@prediction.T-target@ones.T)).mean()

