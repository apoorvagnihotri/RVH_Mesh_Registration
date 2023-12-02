from pathlib import Path

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
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
        # the forward function of SMPLLayer expects rotation matrices instead of axis-angle
        rot_global_orient = axis_angle_to_matrix(self.global_orient)
        rot_body_pose = axis_angle_to_matrix(self.pose.reshape(self.batch_size, -1, 3))

        output = self.layer.forward(
            betas=torch.cat([self.top_betas, self.other_betas], dim=1),
            body_pose=rot_body_pose,
            global_orient=rot_global_orient,
            transl=self.transl,
            **kwargs
        )

        # changing the output of SMPLLayer to match the original SMPL output
        # changing back to axis-angle
        output.global_orient = matrix_to_axis_angle(output.global_orient)
        output.body_pose = matrix_to_axis_angle(output.body_pose).reshape(self.batch_size, -1)
        if kwargs.get("return_full_pose", False):
            output.full_pose = torch.concat([output.global_orient, output.body_pose], dim=1)
        return output
