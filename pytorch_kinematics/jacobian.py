import torch
from pytorch_kinematics import transforms


def calc_jacobian(serial_chain, th, tool=None):
    """
    Return robot Jacobian J in base frame (N,6,DOF) where dot{x} = J dot{q}
    The first 3 rows relate the translational velocities and the
    last 3 rows relate the angular velocities.

    tool is the transformation wrt the end effector; default is identity. If specified, will have to
    specify for each of the N inputs
    """
    if not torch.is_tensor(th):
        th = torch.tensor(th, dtype=serial_chain.dtype, device=serial_chain.device)
    if len(th.shape) <= 1:
        N = 1
        th = th.view(1, -1)
    else:
        N = th.shape[0]
    ndof = th.shape[1]

    j_fl = torch.zeros(
        (N, 6, ndof), dtype=serial_chain.dtype, device=serial_chain.device
    )

    if tool is None:
        cur_transform = (
            transforms.Transform3d(device=serial_chain.device, dtype=serial_chain.dtype)
            .get_matrix()
            .repeat(N, 1, 1)
        )
    else:
        if tool.dtype != serial_chain.dtype or tool.device != serial_chain.device:
            tool = tool.to(
                device=serial_chain.device, copy=True, dtype=serial_chain.dtype
            )
        cur_transform = tool.get_matrix()

    final_pose = serial_chain.forward_kinematics(th, end_only=True).get_matrix()
    final_pos = final_pose[:, :3, 3]

    cnt = 0
    for f in serial_chain._serial_frames:
        if f.joint.joint_type == "fixed":
            tensor = th[:, 0].view(N, 1)
        else:
            tensor = th[:, cnt].view(N, 1)
        cur_frame_transform = f.get_transform(tensor).get_matrix()
        cur_transform = cur_transform @ cur_frame_transform
        if f.joint.joint_type == "revolute":
            delta = final_pos - cur_transform[:, :3, 3]
            angles = f.joint.axis @ cur_transform[:, :3, :3]
            position = torch.cross(angles, delta)
            j_fl[:, :, cnt] = torch.cat((position, angles), dim=-1)
            cnt += 1
        elif f.joint.joint_type == "prismatic":
            j_fl[:, :3, cnt] = f.joint.axis @ cur_transform[:, :3, :3]
            cnt += 1

    return j_fl
