import numpy as np
from .strategy import Strategy

# class MarginSampling(Strategy):
#     def __init__(self, dataset, net):
#         super(MarginSampling, self).__init__(dataset, net)

#     def query(self, n):
#         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
#         probs = self.predict_prob(unlabeled_data)
#         probs_sorted, idxs = probs.sort(descending=True)
#         uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
#         return unlabeled_idxs[uncertainties.sort()[1][:n]]
    

import torch

class MarginSampling(Strategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        predictions = self.predict(unlabeled_data)
        uncertainties = self.calculate_uncertainties(predictions)
        return unlabeled_idxs[uncertainties.sort()[1][:n]]

    def calculate_uncertainties(self, predictions):
        # Here you need to define how to calculate uncertainty for regression.
        # One common approach is to use prediction variance.
        # You can use the variance of predictions across different models
        # or the variance of predictions from an ensemble of models.
        # For simplicity, let's assume prediction variance for now.
        # You can replace this with more sophisticated uncertainty estimation methods if needed.
        prediction_variances = torch.var(predictions, dim=1)
        return prediction_variances

