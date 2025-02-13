"""
If code works:
    Author: Bharat
else:
    Author: Anonymous
"""
import pickle as pkl
from os.path import join

import numpy as np
import torch


def get_prior(
    model_root: str, gender: str = "male", precomputed: bool = True, device: str = "cuda:0"
):
    if precomputed:
        prior = Prior(sm=None, model_root=model_root, device=device)
        return prior["Generic"]
    else:
        raise NotImplemented


class ThMahalanobis:
    def __init__(self, mean, prec, prefix, end=66, device="cuda:0"):
        """Allows to compute the Mahalanobis distance."""
        self.mean = (
            torch.tensor(mean.astype("float32"), requires_grad=False).unsqueeze(axis=0).to(device)
        )
        self.prec = torch.tensor(prec.astype("float32"), requires_grad=False).to(device)
        self.prefix = prefix
        self.end = end

    def __call__(self, pose, prior_weight=1.0):
        """
        :param pose: Batch x pose_dims
        :return: weighted L2 distance of the N pose parameters, where N = 72 - prefix for SMPL model, for smplh, only compute from 3 to 66
        """
        # return (pose[:, self.prefix:] - self.mean)*self.prec
        temp = pose[:, self.prefix : self.end] - self.mean
        temp2 = torch.matmul(temp, self.prec) * prior_weight
        return (temp2 * temp2).sum(dim=1)


class Prior:
    def __init__(self, sm, model_root=None, prefix=3, end=66, device="cuda:0"):
        """Allows to create a prior for the pose parameters of the SMPL model.

        Args:
            sm: ???
            model_root: path to the folder containing the model files
            prefix: number of parameters to ignore in the pose vector
            end: number of parameters to ignore in the pose vector

        Note: end=66 for smplh, 69 for smpl
        """

        self.prefix = prefix
        self.device = device
        self.end = end
        if sm is not None:
            # Compute mean and variance based on the provided poses
            self.pose_subjects = sm.pose_subjects
            # if 'CAESAR' in name or 'Tpose' in name or 'ReachUp' in name]
            all_samples = [
                p[prefix:]
                for qsub in self.pose_subjects
                for name, p in zip(qsub["pose_fnames"], qsub["pose_parms"])
            ]
            self.priors = {"Generic": self.create_prior_from_samples(all_samples)}
            # TODO: figure out if end is actually not supposed to be used here.
        else:
            # Load pre-computed mean and variance, this prior is adapted for smplh model
            file = join(model_root, "priors", "body_prior.pkl")
            dat = pkl.load(open(file, "rb"))
            self.priors = {
                "Generic": ThMahalanobis(
                    dat["mean"], dat["precision"], self.prefix, self.end, self.device
                )
            }

    def create_prior_from_samples(self, samples):
        from sklearn.covariance import GraphicalLassoCV

        model = GraphicalLassoCV()
        model.fit(np.asarray(samples))
        return ThMahalanobis(
            np.asarray(samples).mean(axis=0),
            np.linalg.cholesky(model.precision_),
            self.prefix,
            self.device,
        )

    def __getitem__(self, pid):
        if pid not in self.priors:
            samples = [
                p[self.prefix :]
                for qsub in self.pose_subjects
                for name, p in zip(qsub["pose_fnames"], qsub["pose_parms"])
                if pid in name.lower()
            ]
            self.priors[pid] = (
                self.priors["Generic"]
                if len(samples) < 3
                else self.create_prior_from_samples(samples)
            )

        return self.priors[pid]
