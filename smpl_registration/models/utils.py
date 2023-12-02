from typing import Union

from smplx.body_models import SMPL

from lib.body_objectives import torch_pose_obj_data
from lib.torch_functions import batch_sparse_dense_matmul

from .split_smpl import SplitSMPL


def get_landmarks(smpl: Union[SplitSMPL, SMPL]):
    weight_mats = torch_pose_obj_data(smpl.model_root, batch_size=smpl.batch_size)
    body25_reg_torch, face_reg_torch, hand_reg_torch = weight_mats

    output = smpl.forward(return_full_pose=True)
    verts = output.vertices

    J = batch_sparse_dense_matmul(body25_reg_torch, verts)
    face = batch_sparse_dense_matmul(face_reg_torch, verts)
    hands = batch_sparse_dense_matmul(hand_reg_torch, verts)

    return J, face, hands
