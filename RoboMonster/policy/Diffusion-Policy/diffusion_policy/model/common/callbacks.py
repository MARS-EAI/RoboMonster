import itertools
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from lightning.pytorch.callbacks.callback import Callback
from torch.optim.swa_utils import AveragedModel


class ModelAveragingCallback(Callback):
    def __init__(self, device, avg_fn, update_after_steps=-1):
        self._device = device
        self._avg_fn = avg_fn
        self._averaged_model = None
        self._latest_update_step = update_after_steps

    def on_fit_start(self, trainer, pl_module) -> None:
        device = self._device or pl_module.device
        self._averaged_model = AveragedModel(model=pl_module, device=device, avg_fn=self._avg_fn, use_buffers=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > self._latest_update_step:
            self._averaged_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    def on_fit_end(self, trainer, pl_module):
        average_params = itertools.chain(self._averaged_model.module.parameters(), self._averaged_model.module.buffers())
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            current_param.data.copy_(average_param.data)

    def on_validation_epoch_start(self, trainer, pl_module):
        if self._averaged_model is not None:
            self._swap_models(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._averaged_model is not None:
            self._swap_models(pl_module)

    def state_dict(self):
        return {"latest_update_step": self._latest_update_step}

    def load_state_dict(self, state_dict):
        self._latest_update_step = state_dict["latest_update_step"]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        average_state = self._averaged_model.state_dict()
        checkpoint["current_state_dict"] = checkpoint["state_dict"]
        checkpoint["state_dict"] = OrderedDict({
            name[7:]: value for name, value in average_state.items() if name.startswith("module.")
        })
        # checkpoint["model_averaging_state"] = {
        #     name: value for name, value in average_state.items() if not name.startswith("module.")
        # }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if ("current_state_dict" in checkpoint) and ("model_averaging_state" in checkpoint):
            average_state = {"module." + name: value for name, value in checkpoint["state_dict"].items()}
            average_state |= checkpoint["model_averaging_state"]
            self._averaged_model.load_state_dict(average_state)
            checkpoint["state_dict"] = checkpoint["current_state_dict"]
        else:
            self._averaged_model.module.load_state_dict(deepcopy(checkpoint['state_dict']), strict=False)

    def _swap_models(self, pl_module):
        average_params = itertools.chain(self._averaged_model.module.parameters(), self._averaged_model.module.buffers())
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            tmp = average_param.data.clone()
            average_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)


class SaveConfigCallback(Callback):
    def __init__(self, cfg):
        self.cfg = cfg

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 保存 cfg 到 checkpoint 字典中
        checkpoint['cfg'] = self.cfg
