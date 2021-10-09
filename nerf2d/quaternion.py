"""Component providing modules and functions for quaternion pytorch maths."""

import torch


def _make_elementary_quat(axis: int, angles: torch.Tensor):
    angles = 0.5 * angles
    zeros = torch.zeros_like(angles)
    values = [zeros] * 4
    values[3] = torch.cos(angles)
    values[axis] = torch.sin(angles)
    return torch.stack(values, -1)


COMPOSE = torch.FloatTensor([
    [0, 0, 0, 1,
     0, 0, 1, 0,
     0, -1, 0, 0,
     1, 0, 0, 0],
    [0, 0, -1, 0,
     0, 0, 0, 1,
     1, 0, 0, 0,
     0, 1, 0, 0],
    [0, 1, 0, 0,
     -1, 0, 0, 0,
     0, 0, 0, 1,
     0, 0, 1, 0],
    [-1, 0, 0, 0,
     0, -1, 0, 0,
     0, 0, -1, 0,
     0, 0, 0, 1]
])


def qcompose(r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Composes two quaternions."""
    rs = torch.bmm(r.unsqueeze(-1), s.unsqueeze(-2)).reshape(-1, 16, 1)
    transform = COMPOSE.to(dtype=r.dtype, device=r.device)
    transform = transform.reshape(1, 4, 16).expand(r.shape[0], -1, -1)
    result = torch.bmm(transform, rs)
    return result.squeeze(-1)


MULTIPLY = torch.FloatTensor([
    [0, 0, 0, 1,
     0, 0, -1, 0,
     0, 1, 0, 0,
     1, 0, 0, 0],
    [0, 0, 1, 0,
     0, 0, 0, 1,
     -1, 0, 0, 0,
     0, 1, 0, 0],
    [0, -1, 0, 0,
     1, 0, 0, 0,
     0, 0, 0, 1,
     0, 0, 1, 0],
    [-1, 0, 0, 0,
     0, -1, 0, 0,
     0, 0, -1, 0,
     0, 0, 0, 1]
])


def qmultiply(r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Performs quaternion multiplication."""
    rs = torch.bmm(r.unsqueeze(-1), s.unsqueeze(-2)).reshape(-1, 16, 1)
    transform = MULTIPLY.to(dtype=r.dtype, device=r.device)
    transform = transform.reshape(1, 4, 16).expand(r.shape[0], -1, -1)
    result = torch.bmm(transform, rs)
    return result.squeeze(-1)


def q_from_point(point: torch.Tensor) -> torch.Tensor:
    """Converts a point to a quaternion."""
    zeros = torch.zeros((point.shape[0], 1), dtype=point.dtype, device=point.device)
    p = torch.cat([point, zeros], dim=-1)
    return p


def q_to_point(p: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion to a point."""
    return p[..., :3]


INVERSE = torch.FloatTensor([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


def qinverse(q: torch.Tensor) -> torch.Tensor:
    """Performs quaternion inversion."""
    transform = INVERSE.to(dtype=q.dtype, device=q.device)
    transform = transform.reshape(1, 4, 4).expand(q.shape[0], -1, -1)
    q_inv = torch.bmm(transform, q.unsqueeze(-1)).squeeze(-1)
    return q_inv


def qrotate(q: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Rotates the points using the provided quaternions."""
    assert len(q.shape) == len(point.shape) == 2
    assert q.shape[-1] == 4
    assert point.shape[-1] == 3

    p = q_from_point(point)
    q_inv = qinverse(q)
    p_rot = qmultiply(q_inv, qmultiply(p, q))
    return q_to_point(p_rot)


def q_from_euler_angles(angles: torch.Tensor) -> torch.Tensor:
    """Creates quaternions from the provided Euler angles."""
    [x, y, z] = angles.permute(1, 0).split(1)
    quat = _make_elementary_quat(0, x[0])
    quat = qcompose(quat, _make_elementary_quat(1, y[0]))
    quat = qcompose(quat, _make_elementary_quat(2, z[0]))
    return quat


TO_EULER_ANGLES = torch.FloatTensor([
    [0, 0, 0, 1,
     0, 0, -1, 0,
     0, -1, 0, 0,
     1, 0, 0, 0],
    [-1, 0, 0, 0,
     0, -1, 0, 0,
     0, 0, 1, 0,
     0, 0, 0, 1],
    [0, 0, 1, 0,
     0, 0, 0, 1,
     1, 0, 0, 0,
     0, 1, 0, 0],
    [0, -1, 0, 0,
     -1, 0, 0, 0,
     0, 0, 0, 1,
     0, 0, 1, 0],
    [1, 0, 0, 0,
     0, -1, 0, 0,
     0, 0, -1, 0,
     0, 0, 0, 1]
])


def q_to_euler_angles(q: torch.Tensor) -> torch.Tensor:
    """Converts quaternions to euler angles."""
    qs = torch.bmm(q.unsqueeze(-1), q.unsqueeze(-2)).reshape(-1, 16, 1)
    transform = TO_EULER_ANGLES.to(dtype=q.dtype, device=q.device)
    transform = transform.reshape(1, 5, 16).expand(q.shape[0], -1, -1)
    result = torch.bmm(transform, qs).squeeze(-1)
    r11, r12, r21, r31, r32 = torch.split(result, 1, dim=-1)
    return torch.cat([torch.atan2(r11, r12), torch.asin(r21), torch.atan2(r31, r32)], dim=-1)


TO_MATRIX = torch.FloatTensor([
    [1, 0, 0, 0,
     0, -1, 0, 0,
     0, 0, -1, 0,
     0, 0, 0, 1],
    [0, 1, 0, 0,
     1, 0, 0, 0,
     0, 0, 0, -1,
     0, 0, -1, 0],
    [0, 0, 1, 0,
     0, 0, 0, 1,
     1, 0, 0, 0,
     0, 1, 0, 0],
    [0, 1, 0, 0,
     1, 0, 0, 0,
     0, 0, 0, 1,
     0, 0, 1, 0],
    [-1, 0, 0, 0,
     0, 1, 0, 0,
     0, 0, -1, 0,
     0, 0, 0, 1],
    [0, 0, 0, -1,
     0, 0, 1, 0,
     0, 1, 0, 0,
     -1, 0, 0, 0],
    [0, 0, 1, 0,
     0, 0, 0, -1,
     1, 0, 0, 0,
     0, -1, 0, 0],
    [0, 0, 0, 1,
     0, 0, 1, 0,
     0, 1, 0, 0,
     1, 0, 0, 0],
    [-1, 0, 0, 0,
     0, -1, 0, 0,
     0, 0, 1, 0,
     0, 0, 0, 1]
])


def q_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion to a rotation matrix."""
    q2 = torch.bmm(q.unsqueeze(-1), q.unsqueeze(-2)).reshape(-1, 16, 1)
    transform = TO_MATRIX.to(dtype=q.dtype, device=q.device)
    transform = transform.reshape(1, 9, 16).expand(q.shape[0], -1, -1)
    result = torch.bmm(transform, q2)
    return result.squeeze(-1).reshape(-1, 3, 3)
