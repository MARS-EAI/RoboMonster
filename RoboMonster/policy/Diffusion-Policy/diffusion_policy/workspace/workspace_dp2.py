import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, random_split
from torch.optim.swa_utils import get_ema_avg_fn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import LightningModule


if __name__ == "__main__":
    import sys
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
from diffusion_policy.model.common.callbacks import ModelAveragingCallback, SaveConfigCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name='config'
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    output_dir = pathlib.Path(HydraConfig.get().run.dir)
    # set seed
    seed = cfg.seed
    seed_everything(seed)
    # configure model
    model: LightningModule = hydra.utils.instantiate(cfg.policy)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.95), len(dataset) - int(len(dataset)*0.95)])
    train_dataloader = DataLoader(train_dataset, **cfg.dataloader.train)
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader.val)

    model.set_normalizer(dataset.get_normalizer())

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        hydra.utils.instantiate(cfg.checkpoint, dirpath=output_dir / 'checkpoints'),
        ModelAveragingCallback(None, get_ema_avg_fn(0.9), cfg.ema.update_after_steps),
        SaveConfigCallback(OmegaConf.to_container(cfg, resolve=True))
    ]

    logger = WandbLogger(
        save_dir=output_dir,
        **cfg.logging,
    )

    trainer = Trainer(
        **cfg.trainer,
        strategy="ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto",
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    main()
