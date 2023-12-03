from pathlib import Path

import torch
from smplx.body_models import SMPLLayer
from smplx.utils import SMPLOutput
from smplx.lbs import batch_rodrigues


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
        return [self.transl, self.top_betas, self.global_orient, self.pose]     # Hands Dont get Optimised here?

    def aa2rot(self, pose : torch.Tensor) -> torch.Tensor:
        # Converts Axis-Angle to Rotation Matrix such that Pose can be Passed to Layer

        bone_count = int(pose.shape[1] / 3)
        rot_mats = []

        for idx in range(bone_count):
            axis_ang = pose[:, idx * 3 : (idx + 1) * 3] # (B x 3) 
            rot_mat = batch_rodrigues(axis_ang)         # (B x 3 x 3) - 3x3 rotation matrix 
            rot_mat = torch.reshape(rot_mat, (1, -1))   # (B x 9)   
            rot_mats.append(rot_mat)                    # (bone_count x B x 9)

        rot_mats = torch.cat(rot_mats, dim=1)               # (B x bone_count * 9)
        return rot_mats

    def forward(self, **kwargs) -> SMPLOutput:
        return self.layer.forward(
            betas=torch.cat([self.top_betas, self.other_betas], dim=1),
            # body_pose=self.pose[:, 3:],               - Global Orient is Not in Here + Hands Should be their Own Thing 
            body_pose=self.aa2rot(self.pose),
            global_orient=self.aa2rot(self.global_orient),
            transl=self.transl,
            **kwargs
            )
