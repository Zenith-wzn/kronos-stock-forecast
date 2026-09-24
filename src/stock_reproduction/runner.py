from __future__ import annotations
import argparse, json, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from .audit import audit_inputs
from .common import ROOT, atomic_json, ensure_layout, set_determinism
from .evaluate import evaluate
from .models import corrected_pairwise_rank_loss, load_master, load_sspt, upstream_sspt_rank_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["master", "sspt"], required=True)
    parser.add_argument("--stage", choices=["prepare", "smoke", "train", "evaluate", "all"], default="all")
    parser.add_argument("--device", default=None, help="cpu, cuda:0, or another torch device")
    parser.add_argument("--pretrain-checkpoint", default=None, help="SSPT classification pretraining checkpoint")
    parser.add_argument("--skip-pretrain", action="store_true", help="reuse --pretrain-checkpoint and skip SSPT pretraining")
    parser.add_argument("--output-dir", default=None, help="directory for this run's results and weights")
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    return parser.parse_args()

def read_config(home: Path) -> dict: return json.loads((home/"config.json").read_text(encoding="utf-8"))

def record_dependencies(home: Path) -> None:
    text = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (home/"requirements.lock.txt").write_text(text, encoding="utf-8")

def apply_overrides(config: dict, args) -> dict:
    config = json.loads(json.dumps(config))
    if args.device is not None:
        config["device"] = args.device
    if args.pretrain_checkpoint is not None:
        config["training"]["pretrain_checkpoint"] = args.pretrain_checkpoint
    if args.skip_pretrain:
        config["training"]["skip_pretrain"] = True
    if args.output_dir is not None:
        config["_output_dir"] = args.output_dir
    if args.pretrain_epochs is not None:
        config["training"]["pretrain_epochs"] = args.pretrain_epochs
    if args.finetune_epochs is not None:
        config["training"]["finetune_epochs"] = args.finetune_epochs
    return config

def smoke(home: Path, model: str, config: dict) -> dict:
    set_determinism(config["seed"], config["threads"]); started = time.perf_counter()
    device_name = config.get("device", "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
    device = torch.device(device_name)
    if model == "master":
        network = load_master(home, config).to(device); inputs = torch.randn(32, 90, 221, device=device); targets = torch.randn(32, device=device); predictions = network(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
    else:
        network = load_sspt(home, config).to(device); inputs = torch.randn(32, 90, 25, device=device); targets = torch.linspace(-1, 1, 32, device=device); predictions = network(inputs).squeeze(-1)
        upstream = float(upstream_sspt_rank_loss(predictions.detach(), targets)); corrected = corrected_pairwise_rank_loss(predictions, targets)
        if corrected <= 0: raise ValueError("Corrected ranking loss test did not detect reversed ordering")
        loss = torch.nn.functional.mse_loss(predictions, targets) + config["training"]["ranking_loss_alpha"]*corrected
    loss.backward()
    if not np.isfinite(float(loss)): raise ValueError("Non-finite smoke loss")
    output_home = Path(config.get("_output_dir", home)); (output_home/"weights").mkdir(parents=True, exist_ok=True); (output_home/"results").mkdir(parents=True, exist_ok=True)
    path = output_home/"weights/smoke_gpu.pt" if device.type == "cuda" else output_home/"weights/smoke.pt"; torch.save(network.state_dict(), path)
    clone = (load_master(home, config) if model == "master" else load_sspt(home, config)).to(device); clone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); clone.eval(); network.eval()
    with torch.inference_mode():
        if not torch.allclose(network(inputs), clone(inputs), rtol=1e-5, atol=1e-6): raise ValueError("Checkpoint reload mismatch")
    result = {"status": "complete", "model": model, "loss": float(loss), "seconds": time.perf_counter()-started, "parameters": sum(p.numel() for p in network.parameters())}
    if model == "sspt": result["ranking_loss_audit"] = {"upstream_on_reversed_example": upstream, "corrected_on_reversed_example": float(corrected), "finding": "Upstream implementation mixes predicted values into the target-difference matrix; aligned runner uses the standard target pairwise difference."}
    result["device"] = str(device)
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    atomic_json(output_home/"results/smoke.json", result); return result

def train(home: Path, model: str, audit: dict, config: dict) -> None:
    if audit["status"] != "ready":
        payload = {"status": "blocked", "reason": "Required external data is unavailable or unverifiable; training was not silently simplified.", "blockers": audit["blockers"]}
        atomic_json(home/"results/training_status.json", payload); raise RuntimeError(payload["reason"])
    if model == 'sspt':
        from .sspt_aligned import run
        run(home, config); return
    raise NotImplementedError("MASTER training remains blocked until its official market and VWAP inputs are available")

def main():
    args = parse_args(); home = ROOT/args.model; ensure_layout(home); config = apply_overrides(read_config(home), args)
    try:
        audit = audit_inputs(home, args.model, config); record_dependencies(home)
        if args.stage == "prepare": print(json.dumps(audit, ensure_ascii=False, indent=2)); return
        if args.stage in ("smoke", "all"): print(json.dumps(smoke(home, args.model, config), ensure_ascii=False, indent=2))
        if args.stage in ("train", "all"): train(home, args.model, audit, config)
        if args.stage == "evaluate": print(json.dumps(evaluate(home, args.model, config["seed"]), ensure_ascii=False, indent=2))
    except Exception as error:
        atomic_json(home/"results/failure.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}); raise

if __name__ == "__main__": main()
