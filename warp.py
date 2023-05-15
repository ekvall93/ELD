import torch 
from typing import Tuple

import torch
import torch.nn as nn
import ipdb
from kornia.utils import create_meshgrid
import torch.nn.functional as F

class TPS:
    def __init__(self, warp_type):
        assert warp_type in ["tps", "affine"], "Wrong type"
        self.use_tps = True if warp_type == "tps" else False
        print(warp_type)
        print(self.use_tps)
        
    def __pair_square_euclidean(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        r"""Compute the pairwise squared euclidean distance matrices :math:`(B, N, M)` between two tensors
        with shapes (B, N, C) and (B, M, C)."""
        # ||t1-t2||^2 = (t1-t2)^T(t1-t2) = t1^T*t1 + t2^T*t2 - 2*t1^T*t2
        t1_sq: torch.Tensor = tensor1.mul(tensor1).sum(dim=-1, keepdim=True)
        t2_sq: torch.Tensor = tensor2.mul(tensor2).sum(dim=-1, keepdim=True).transpose(1, 2)
        t1_t2: torch.Tensor = tensor1.matmul(tensor2.transpose(1, 2))
        square_dist: torch.Tensor = -2 * t1_t2 + t1_sq + t2_sq
        square_dist = square_dist.clamp(min=0)  # handle possible numerical errors
        return square_dist

    def _pair_square_euclidean(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        r"""Compute the pairwise squared euclidean distance matrices :math:`(B, N, M)` between two tensors
        with shapes (B, N, C) and (B, M, C)."""
        # ||t1-t2||^2 = (t1-t2)^T(t1-t2) = t1^T*t1 + t2^T*t2 - 2*t1^T*t2
        square_dist =  torch.cdist(tensor1, tensor2)
        square_dist = square_dist.clamp(min=0)  # handle possible numerical errors
        return square_dist


    def kernel_distance(self, squared_distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        r"""Compute the TPS kernel distance function: :math:`r^2 log(r)`, where `r` is the #euclidean distance.
        Since :math:`\log(r) = 1/2 \log(r^2)`, this function takes the squared distance matrix and calculates
        :math:`0.5 r^2 log(r^2)`."""
        
        loss = 0.5 * squared_distances * squared_distances.add(eps).log()
        return loss
    
    def warp_points_tps(self, points_src: torch.Tensor, kernel_centers: torch.Tensor,
                        kernel_weights: torch.Tensor, affine_weights: torch.Tensor) -> torch.Tensor:
        
        # f_{x|y}(v) = a_0 + [a_x a_y].v + \sum_i w_i * U(||v-u_i||)
        pair_distance: torch.Tensor = self._pair_square_euclidean(points_src, kernel_centers)
        b, d1, d2 = pair_distance.shape

        #l = torch.cat([points_src, pair_distance],2)
        k_matrix: torch.Tensor = self.kernel_distance(pair_distance)

        
        if self.use_tps:
            warped: torch.Tensor = (
                (k_matrix[..., None].mul(kernel_weights[:, None]).sum(-2) +
                points_src[..., None].mul(affine_weights[:, None, 1:]).sum(-2) +
                affine_weights[:, None, 0])
            )
        else:
            warped: torch.Tensor = (
            points_src[..., None].mul(affine_weights[:, None, 1:]).sum(-2) +
            affine_weights[:, None, 0])
        

        
        return warped

    def get_tps_transform(self, points_src: torch.Tensor, points_dst: torch.Tensor):
        r"""Compute the TPS transform parameters that warp source points to target points.

        The input to this function is a tensor of :math:`(x, y)` source points :math:`(B, N, 2)` and a corresponding
        tensor of target :math:`(x, y)` points :math:`(B, N, 2)`.

        Args:
            points_src (torch.Tensor): batch of source points :math:`(B, N, 2)` as :math:`(x, y)` coordinate vectors.
            points_dst (torch.Tensor): batch of target points :math:`(B, N, 2)` as :math:`(x, y)` coordinate vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: :math:`(B, N, 2)` tensor of kernel weights and :math:`(B, 3, 2)`
                tensor of affine weights. The last dimension contains the x-transform and y-transform weights
                as seperate columns.

        Example:
            >>> points_src = torch.rand(1, 5, 2)
            >>> points_dst = torch.rand(1, 5, 2)
            >>> kernel_weights, affine_weights = get_tps_transform(points_src, points_dst)

        .. note::
            This function is often used in conjuntion with :func:`warp_points_tps`, :func:`warp_image_tps`.
        """
        if not isinstance(points_src, torch.Tensor):
            raise TypeError(f"Input points_src is not torch.Tensor. Got {type(points_src)}")

        if not isinstance(points_dst, torch.Tensor):
            raise TypeError(f"Input points_dst is not torch.Tensor. Got {type(points_dst)}")

        if not len(points_src.shape) == 3:
            raise ValueError(f"Invalid shape for points_src, expected BxNx2. Got {points_src.shape}")

        if not len(points_dst.shape) == 3:
            raise ValueError(f"Invalid shape for points_dst, expected BxNx2. Got {points_dst.shape}")

        device, dtype = points_src.device, points_src.dtype
        batch_size, num_points = points_src.shape[:2]

        # set up and solve linear system
        # [K   P] [w] = [dst]
        # [P^T 0] [a]   [ 0 ]
        #pair_distance: torch.Tensor = self._pair_square_euclidean(points_src, points_dst)
        pair_distance: torch.Tensor = self._pair_square_euclidean(points_src, points_src)
        #ipdb.set_trace()
        n_pts = points_src.size(1)
        #print(n_pts)
        k_matrix = self.kernel_distance(pair_distance)
        mask = torch.linalg.matrix_rank(k_matrix) != n_pts

        
        k_matrix[mask] = k_matrix[mask] + torch.eye(n_pts,n_pts).cuda()[None].repeat(sum(mask),1,1) * 0.001


        zero_mat: torch.Tensor = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        one_mat: torch.Tensor = torch.ones(batch_size, num_points, 1, device=device, dtype=dtype)
        dest_with_zeros: torch.Tensor = torch.cat((points_dst, zero_mat[:, :, :2]), 1)
        p_matrix: torch.Tensor = torch.cat((one_mat, points_src), -1)

        

        p_matrix_t: torch.Tensor = torch.cat((p_matrix, zero_mat), 1).transpose(1, 2)


        l_matrix = torch.cat((k_matrix, p_matrix), -1)
        l_matrix = torch.cat((l_matrix, p_matrix_t), 1)
        weights, _ = torch.solve(dest_with_zeros, l_matrix)

        if not self.use_tps:
            affine_weights = torch.linalg.pinv(p_matrix) @ points_dst
            rbf_weights = affine_weights
            return (rbf_weights, affine_weights)

        rbf_weights = weights[:, :-3]
        affine_weights = weights[:, -3:]


        

        return (rbf_weights, affine_weights)



    def kernel_distance(self, squared_distances: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        loss = 0.5 * squared_distances * squared_distances.add(eps).log()
        return loss

    def warp_image_tps(self, image: torch.Tensor, kernel_centers: torch.Tensor, kernel_weights: torch.Tensor,
                    affine_weights: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
        r"""Warp an image tensor according to the thin plate spline transform defined by kernel centers,
        kernel weights, and affine weights.

        The transform is applied to each pixel coordinate in the output image to obtain a point in the input
        image for interpolation of the output pixel. So the TPS parameters should correspond to a warp from
        output space to input space.

        The input `image` is a :math:`(B, C, H, W)` tensor. The kernel centers, kernel weight and affine weights
        are the same as in `warp_points_tps`.

        Args:
            image (torch.Tensor): input image tensor :math:`(B, C, H, W)`.
            kernel_centers (torch.Tensor): kernel center points :math:`(B, K, 2)`.
            kernel_weights (torch.Tensor): tensor of kernl weights :math:`(B, K, 2)`.
            affine_weights (torch.Tensor): tensor of affine weights :math:`(B, 3, 2)`.
            align_corners (bool): interpolation flag used by `grid_sample`. Default: False.

        Returns:
            torch.Tensor: warped image tensor :math:`(B, C, H, W)`.

        Example:
            >>> points_src = torch.rand(1, 5, 2)
            >>> points_dst = torch.rand(1, 5, 2)
            >>> image = torch.rand(1, 3, 32, 32)
            >>> # note that we are getting the reverse transform: dst -> src
            >>> kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
            >>> warped_image = warp_image_tps(image, points_src, kernel_weights, affine_weights)

        .. note::
            This function is often used in conjuntion with :func:`get_tps_transform`.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(kernel_centers, torch.Tensor):
            raise TypeError(f"Input kernel_centers is not torch.Tensor. Got {type(kernel_centers)}")

        if not isinstance(kernel_weights, torch.Tensor):
            raise TypeError(f"Input kernel_weights is not torch.Tensor. Got {type(kernel_weights)}")

        if not isinstance(affine_weights, torch.Tensor):
            raise TypeError(f"Input affine_weights is not torch.Tensor. Got {type(affine_weights)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid shape for image, expected BxCxHxW. Got {image.shape}")

        if not len(kernel_centers.shape) == 3:
            raise ValueError(f"Invalid shape for kernel_centers, expected BxNx2. Got {kernel_centers.shape}")

        if not len(kernel_weights.shape) == 3:
            raise ValueError(f"Invalid shape for kernel_weights, expected BxNx2. Got {kernel_weights.shape}")

        if not len(affine_weights.shape) == 3:
            raise ValueError(f"Invalid shape for affine_weights, expected BxNx2. Got {affine_weights.shape}")

        device, dtype = image.device, image.dtype
        batch_size, _, h, w = image.shape
        coords: torch.Tensor = create_meshgrid(h, w, device=device).to(dtype=dtype)
        coords = coords.reshape(-1, 2).expand(batch_size, -1, -1)
        warped: torch.Tensor = self.warp_points_tps(coords, kernel_centers, kernel_weights, affine_weights)
        warped = warped.view(-1, h, w, 2)
        warped_image: torch.Tensor = nn.functional.grid_sample(image, warped, align_corners=align_corners)
        return warped_image

    def warp_image_tps_grid(self, image: torch.Tensor, kernel_centers: torch.Tensor, kernel_weights: torch.Tensor,
                    affine_weights: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
        r"""Warp an image tensor according to the thin plate spline transform defined by kernel centers,
        kernel weights, and affine weights.

        The transform is applied to each pixel coordinate in the output image to obtain a point in the input
        image for interpolation of the output pixel. So the TPS parameters should correspond to a warp from
        output space to input space.

        The input `image` is a :math:`(B, C, H, W)` tensor. The kernel centers, kernel weight and affine weights
        are the same as in `warp_points_tps`.

        Args:
            image (torch.Tensor): input image tensor :math:`(B, C, H, W)`.
            kernel_centers (torch.Tensor): kernel center points :math:`(B, K, 2)`.
            kernel_weights (torch.Tensor): tensor of kernl weights :math:`(B, K, 2)`.
            affine_weights (torch.Tensor): tensor of affine weights :math:`(B, 3, 2)`.
            align_corners (bool): interpolation flag used by `grid_sample`. Default: False.

        Returns:
            torch.Tensor: warped image tensor :math:`(B, C, H, W)`.

        Example:
            >>> points_src = torch.rand(1, 5, 2)
            >>> points_dst = torch.rand(1, 5, 2)
            >>> image = torch.rand(1, 3, 32, 32)
            >>> # note that we are getting the reverse transform: dst -> src
            >>> kernel_weights, affine_weights = get_tps_transform(points_dst, points_src)
            >>> warped_image = warp_image_tps(image, points_src, kernel_weights, affine_weights)

        .. note::
            This function is often used in conjuntion with :func:`get_tps_transform`.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(kernel_centers, torch.Tensor):
            raise TypeError(f"Input kernel_centers is not torch.Tensor. Got {type(kernel_centers)}")

        if not isinstance(kernel_weights, torch.Tensor):
            raise TypeError(f"Input kernel_weights is not torch.Tensor. Got {type(kernel_weights)}")

        if not isinstance(affine_weights, torch.Tensor):
            raise TypeError(f"Input affine_weights is not torch.Tensor. Got {type(affine_weights)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid shape for image, expected BxCxHxW. Got {image.shape}")

        if not len(kernel_centers.shape) == 3:
            raise ValueError(f"Invalid shape for kernel_centers, expected BxNx2. Got {kernel_centers.shape}")

        if not len(kernel_weights.shape) == 3:
            raise ValueError(f"Invalid shape for kernel_weights, expected BxNx2. Got {kernel_weights.shape}")

        if not len(affine_weights.shape) == 3:
            raise ValueError(f"Invalid shape for affine_weights, expected BxNx2. Got {affine_weights.shape}")

        device, dtype = image.device, image.dtype
        batch_size, _, h, w = image.shape
        coords: torch.Tensor = create_meshgrid(h, w, device=device).to(dtype=dtype)
        coords = coords.reshape(-1, 2).expand(batch_size, -1, -1)
        warped: torch.Tensor = self.warp_points_tps(coords, kernel_centers, kernel_weights, affine_weights)
        warped = warped.view(-1, h, w, 2)
        return warped

    def normalize(self, src_pts, dts_pts, pts_to_map=None):
        if isinstance(pts_to_map, torch.Tensor):
            return src_pts / 63.5 - 1 , dts_pts / 63.5 - 1, pts_to_map / 63.5 - 1
        else:
            return src_pts / 63.5 - 1 , dts_pts / 63.5 - 1


    def unnormalize(self, pts):
        return (pts + 1) * 63.5
        
       
    def warp_img(self,img, src, dst, norm=True):
        if norm:
            src, dst = self.normalize(src, dst)
        src, dst = dst, src
        rbf_w, aff_w = self.get_tps_transform(src, dst)
        return self.warp_image_tps(img, src, rbf_w, aff_w)

    def warp_pts(self, src, dst, pts_to_map, norm=True, up_norm=True):
        if norm:
            src, dst, pts_to_map = self.normalize(src, dst, pts_to_map)
        rbf_w, aff_w = self.get_tps_transform(src, dst)
        warped_src: torch.Tensor = self.warp_points_tps(pts_to_map, src, rbf_w, aff_w) 
        if up_norm:
            warped_src = self.unnormalize(warped_src)
        return warped_src

import kornia


from kornia.geometry.conversions import normalize_homography
from kornia.utils.helpers import _torch_inverse_cast

from kornia.geometry.linalg import transform_points

class Homo:
    def __init__(self):
        pass 

    """ def warp_pts(self,scr_pts, dst_pts, mapped_points, size=128):

        B, H, W = 128, 128, 128
        h_out, w_out = 128, 128

        M = kornia.geometry.homography.find_homography_dlt(dst_pts, scr_pts)
        dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M, (H, W), (h_out, w_out))  # Bx3x3
        src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)  # Bx3x3
        mapped_sample_pts = transform_points(src_norm_trans_dst_norm, mapped_points)
        
        return mapped_sample_pts """

    def warp_pts(self,scr_pts, dst_pts, mapped_points, size=128):

      
        M = kornia.geometry.homography.find_homography_dlt(dst_pts, scr_pts)
        mapped_sample_pts = transform_points(M, mapped_points)
        
        return mapped_sample_pts

    
    def warp_img(self, img, scr_pts, dst_pts, size=128):
        M = kornia.geometry.homography.find_homography_dlt(scr_pts, dst_pts)
        img_warp,_ = kornia.geometry.warp_perspective(img, M, dsize=(size, size))
        return img_warp.clip(0,1)

import ipdb
class WARP():
    def __init__(self, warp_type):
        assert warp_type in ["tps", "homo", "affine"], f"warp_type is wrong{warp_type}"
        self.warper = TPS(warp_type) if warp_type in ["tps", "affine"] else Homo()

    def warp_img(self, img, scr_pts, dst_pts):
        return self.warper.warp_img(img, scr_pts, dst_pts)
    
    def warp_pts(self, src, dst, pts_to_map):
        return self.warper.warp_pts(src, dst, pts_to_map)
