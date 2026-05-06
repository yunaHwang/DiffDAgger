#!/usr/bin/env python3
import sys
# sys.path.insert(0, "/code/diffusha")   # diffusha imports
# sys.path.insert(0, "/code/DiffDAgger") # DiffDAgger imports

import os
sys.path.insert(0, "/code/diffusha")
sys.path.insert(0, "/code/DiffDAgger")
os.environ["PYTHONPATH"] = "/code/diffusha:/code/DiffDAgger"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path


@hydra.main(
    version_base=None,
    config_path="/code/DiffDAgger/diffdagger/config/sim",
    config_name="blockpush_state.yaml",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    save_dir = Path(cfg.save_file_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # build dataset
    dataset = instantiate(cfg.dataset)

    # save normalizers for later use in diffdagger_loss.py
    torch.save(dataset.normalizers, Path(cfg.normalizers_path))
    print(f"saved normalizers to {cfg.normalizers_path}")

    # build policy
    policy = instantiate(cfg.policy, _recursive_=False).float().to(cfg.device)

    # set normalizers
    policy.set_normalizers(dataset.normalizers)

    # compute train steps
    cfg.train.train_steps = min(
        cfg.epoch * len(dataset) // cfg.train.train_bs,
        cfg.max_train_steps,
    ) + 1
    policy.hparams.num_warmup_steps = min(cfg.train.train_steps // 10, 500)
    print(f"train steps: {cfg.train.train_steps}")

    chkpt_indices = [cfg.train.train_steps - 1]

    # train
    chkpts = policy.train_model(
        cfg,
        dataset,
        chkpt_indices=chkpt_indices,
        chkpt_dir=str(save_dir),
        log_dir=str(save_dir / "lightning_logs"),
    )

    # load best checkpoint and compute diffusion loss threshold
    print("computing diffusion loss threshold from dataset...")
    policy.load(chkpts[-1])
    policy.to(cfg.device)
    policy.get_stats_from_dataset(dataset)

    final_path = str(save_dir / "blockpush_diffdagger_final.pth")
    policy.save(final_path)
    print(f"saved final model to {final_path}")


if __name__ == "__main__":
    main()