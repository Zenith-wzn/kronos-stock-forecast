from __future__ import annotations
import numpy as np
import torch
from .common import COVARIATES, QUANTILES


class Forecaster:
    def __init__(self, model, weights):
        self.name = model
        if model == 'chronos2':
            from chronos import Chronos2Pipeline
            self.pipeline = Chronos2Pipeline.from_pretrained(str(weights), device_map='cpu', torch_dtype=torch.float32, local_files_only=True)
            self.pipeline.model.eval()
        elif model == 'timesfm3':
            from timesfm3 import TimesFM3Forecaster, ModelConfig
            self.pipeline = TimesFM3Forecaster(ModelConfig(checkpoint_path=str(weights), device='cpu', per_core_batch_size=4, local_files_only=True))
            self.pipeline.model.float().eval()
        else:
            raise ValueError(model)
        if any(p.device.type != 'cpu' or (p.is_floating_point() and p.dtype != torch.float32) for p in self.pipeline.model.parameters()):
            raise ValueError('Expected CPU float32 model')

    def predict(self, inputs, batch_size):
        if any(x.shape != (5, 90) for x in inputs):
            raise ValueError('Expected five channels and 90 real observations')
        with torch.inference_mode():
            if self.name == 'chronos2':
                tasks = [{'target': x[0].copy(), 'past_covariates': {name: x[j+1].copy() for j, name in enumerate(COVARIATES)}} for x in inputs]
                q, _ = self.pipeline.predict_quantiles(tasks, prediction_length=10, quantile_levels=QUANTILES,
                     context_length=90, cross_learning=False, batch_size=batch_size * 5)
                # Chronos internal batch size counts target and covariate variates, not stocks.
                result = np.stack([a.detach().cpu().numpy()[0] for a in q])
            else:
                self.pipeline.config.per_core_batch_size = batch_size
                out = list(self.pipeline.predict_batch(contexts=[x[0].copy() for x in inputs], horizon=10,
                    past_only_covariates=[x[1:].copy() for x in inputs], past_future_covariates=None,
                    return_quantiles=True, use_symmetric_averaging=False, make_positive=False,
                    sort_quantiles=False, use_znorm=False, padding_mode='none'))
                result = np.stack([o.quantiles for o in out])
        if result.shape != (len(inputs), 10, 9):
            raise ValueError(f'Unexpected shape: {result.shape}')
        return result
