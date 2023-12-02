from pathlib import Path

import torch
from smplx.body_models import SMPLLayer
from smplx.utils import SMPLOutput


class SplitSMPL(torch.nn.Module):
    def __init__(
        self, model_path: Path, batch_size, gender, betas, pose, transl, split_params_args
    ):
        super().__init__()
        self.model_path = model_path
        self.model_root = model_path.parent
        self.batch_size = batch_size
        self.gender = gender
        self.top_betas = torch.nn.Parameter(betas[:, : split_params_args["top_betas"]])
        self.other_betas = torch.nn.Parameter(betas[:, split_params_args["top_betas"] :])
        self.global_orient = torch.nn.Parameter(pose[:, :3])
        self.pose = torch.nn.Parameter(pose[:, 3:])
        self.transl = torch.nn.Parameter(transl)
        self.layer = SMPLLayer(model_path)

    @property
    def faces_tensor(self) -> torch.Tensor:
        return self.layer.faces_tensor

    @property
    def full_pose(self) -> torch.Tensor:
        return torch.cat([self.global_orient, self.pose], dim=1)

    @property
    def betas(self) -> torch.Tensor:
        return torch.cat([self.top_betas, self.other_betas], dim=1)

    @property
    def first_stage_params(self) -> list[torch.nn.Parameter]:
        return [self.transl, self.top_betas, self.global_orient]

    @property
    def second_stage_params(self) -> list[torch.nn.Parameter]:
        return [self.transl, self.top_betas, self.global_orient, self.pose]

    def forward(self, **kwargs) -> SMPLOutput:
        self.layer.forward(
            betas=torch.cat([self.top_betas, self.other_betas], dim=1),
            body_pose=self.pose[:, 3:],
            global_orient=self.global_orient,
            transl=self.transl,
            **kwargs
        )
