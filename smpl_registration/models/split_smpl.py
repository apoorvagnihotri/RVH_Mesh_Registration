from pathlib import Path
from typing import Any

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from smplx.body_models import SMPL, SMPLLayer
from smplx.utils import SMPLOutput


class SplitSMPL(torch.nn.Module):
    def __init__(
        self,
        model_path: Path,
        batch_size: int,
        gender: str,
        betas: torch.Tensor,
        pose: torch.Tensor,
        transl: torch.Tensor,
        split_params_args: dict[str, Any],
    ):
        """Creates a split SMPL model which allows us to optimize parts of parameters at different
        stages. This class is only useful when we provide pose files. Otherwise we can just use the
        SMPL model from smplx.

        Args:
            model_path (Path): path to the SMPL model
            batch_size (int): batch size
            gender (str): gender of the model
            betas (torch.Tensor): shape init values
            pose (torch.Tensor): pose init values
            transl (torch.Tensor): translation init values
            split_params_args (dict[str, Any]): arguments for splitting the parameters
                one such argument is "top_betas" which specifies the number of betas to be optimized in the first stage.
        """
        super().__init__()
        self.model_path = model_path
        self.model_root = model_path.parent
        self.batch_size = batch_size
        self.gender = gender
        _num_body_vars = (
            SMPL.NUM_BODY_JOINTS - 2
        ) * 3  # remove 2 hand joins with 3 rotation angles
        self.top_betas = torch.nn.Parameter(betas[:, : split_params_args["top_betas"]])
        self.other_betas = torch.nn.Parameter(betas[:, split_params_args["top_betas"] :])
        self.global_orient = torch.nn.Parameter(pose[:, :3])
        self.body_pose = torch.nn.Parameter(pose[:, 3 : _num_body_vars + 3])
        self.hand_pose = torch.nn.Parameter(pose[:, _num_body_vars + 3 :])
        self.transl = torch.nn.Parameter(transl)
        self.layer = SMPLLayer(model_path)

    @property
    def faces_tensor(self) -> torch.Tensor:
        """Returns the faces tensor of the SMPL model."""
        return self.layer.faces_tensor

    @property
    def full_pose(self) -> torch.Tensor:
        """Returns the full pose of the SMPL model."""
        return torch.cat([self.global_orient, self.body_pose], dim=1)

    @property
    def betas(self) -> torch.Tensor:
        """Returns all the betas of the SMPL model."""
        return torch.cat([self.top_betas, self.other_betas], dim=1)

    @property
    def first_stage_params(self) -> list[torch.nn.Parameter]:
        """Returns the parameters to be optimized in the first stage."""
        return [self.transl, self.top_betas, self.global_orient]

    @property
    def second_stage_params(self) -> list[torch.nn.Parameter]:
        """Returns the parameters to be optimized in the second stage."""
        return [self.transl, self.top_betas, self.global_orient, self.body_pose]

    def forward(self, **kwargs) -> SMPLOutput:
        # the forward function of SMPLLayer expects rotation matrices instead of axis-angle
        rot_global_orient = axis_angle_to_matrix(self.global_orient)
        full_pose = torch.cat(
            [self.body_pose, self.hand_pose], dim=1
        )  # doesn't include global orient
        rot_full_pose = axis_angle_to_matrix(full_pose.reshape(self.batch_size, -1, 3))

        output = self.layer.forward(
            betas=torch.cat([self.top_betas, self.other_betas], dim=1),
            body_pose=rot_full_pose,  # this includes hand pose
            global_orient=rot_global_orient,
            transl=self.transl,
            **kwargs
        )

        # changing the output of SMPLLayer to match the original SMPL output
        # changing back to axis-angle
        output.global_orient = matrix_to_axis_angle(output.global_orient)
        output.body_pose = matrix_to_axis_angle(output.body_pose).reshape(self.batch_size, -1)
        if kwargs.get("return_full_pose", False):
            output.full_pose = torch.cat([output.global_orient, output.body_pose], dim=1)
        return output
