#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
import scipy.spatial.transform as transform

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_tensor(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0,1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt.float()

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    # this is openCV coordinate system
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def transform_pcd(pcd, MLMatrix44):
    # transform pointcloud to zero123 coordinate
    # in this case, we just tranfer the pointcloud to the origin
    # designed for BasicPointCloud
    points_xyz = pcd.points
    ones = np.ones((points_xyz.shape[0], 1))
    homo_xyz = np.hstack((points_xyz, ones))
    new_pc_xyz = (MLMatrix44 @ homo_xyz.T).T
    # create a new Basicpcd
    new_pcd = BasicPointCloud(points=new_pc_xyz[:,:3], colors=pcd.colors, normals=pcd.normals)
    return new_pcd

import scipy.stats as stats
def z_score_from_percentage(percentage):
    """
    Returns the z-score for a given percentage in a standard normal distribution.

    :param percentage: The desired percentage of points remaining (e.g., 5 for 5%)
    :return: The corresponding z-score
    """
    # Convert percentage to a proportion
    proportion = percentage / 100

    # Calculate the z-score
    z_score = stats.norm.ppf(1 - proportion)

    return z_score

def interpolate_camera_poses(camera_poses, num_virtual_poses):
    virtual_poses_R = []
    virtual_poses_T = []
    virtual_camera_center = []
    virtual_world_view_transform = []
    virtual_full_proj_transform = []
    for i in range(len(camera_poses)-1):
        pose1_index = i
        pose2_index = i + 1

        pose1 = camera_poses[pose1_index].R
        pose2 = camera_poses[pose2_index].R 
        t1 = camera_poses[pose1_index].T
        t2 = camera_poses[pose2_index].T

        r1_2 = transform.Rotation.from_matrix([pose1,pose2])

        for j in range(num_virtual_poses):
            t = j / num_virtual_poses
            interpolated_r = transform.Slerp([0, 1], r1_2)(t).as_matrix()
            interpolated_t = t1 * (1 - t) + t2 * t
            virtual_poses_R.append(interpolated_r)
            virtual_poses_T.append(interpolated_t)

            world_view_transform = torch.tensor(getWorld2View2(interpolated_r, interpolated_t, \
                        camera_poses[pose1_index].trans, camera_poses[pose1_index].scale)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(camera_poses[pose1_index].znear, camera_poses[pose1_index].zfar, camera_poses[pose1_index].FoVx, camera_poses[pose1_index].FoVy).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            virtual_camera_center.append(camera_center)
            virtual_full_proj_transform.append(full_proj_transform)
            virtual_world_view_transform.append(world_view_transform)


    return virtual_poses_R, virtual_poses_T, virtual_camera_center, virtual_world_view_transform, virtual_full_proj_transform


    