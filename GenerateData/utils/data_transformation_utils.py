import math
import torch
import numpy as np

def get_entry_size(attributes):
    #get gausian point entry size
    #+1 for geodesic distance
    entry_size = sum([4 if attr == "rotation" else 3 if attr in ["xyz", "normals", "scale", "sh"] else 1 for attr in attributes])
    return entry_size +1


def get_masked_entry(attributes, mask_constant):
    #get a masked gaussian point entry
    #used in data augmentation dox max neighbors and dropout

    masked_entry = torch.zeros(1, get_entry_size(attributes), dtype=torch.float32)
    attr_index = 0
    for attr in attributes:
        if attr == "xyz":
            masked_entry[0, attr_index:attr_index+3] = mask_constant
            attr_index += 3
        elif attr == "normals":
            #masked_entry[0, attr_index:attr_index+3] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            attr_index += 3
        elif attr == "opacity":
            masked_entry[0, attr_index] = 0.0
            attr_index += 1
        elif attr == "scale":
            masked_entry[0, attr_index:attr_index+3] = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            attr_index += 3
        elif attr == "rotation":
            # aplly unit quaternion for masking
            masked_entry[0, attr_index:attr_index+4] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            attr_index += 4
        elif attr == "sh":
            masked_entry[0, attr_index:attr_index+3] =  -0.5*2/ math.sqrt(3)
            attr_index += 3
        elif attr == "euclidean_distances":
            masked_entry[0, attr_index] = mask_constant
            attr_index += 1
        #geodesic distances
        masked_entry[0, attr_index] = mask_constant 
    return masked_entry 

def canonical_rotation(A, b=None):
    """
    Compute a canonical rotation matrix that aligns vector 'a' with vector 'b'.
    
    This function computes a 3x3 rotation matrix R such that R @ a points in the direction of b.
    The rotation is "canonical" in the sense that it's constructed systematically using
    the Gram-Schmidt process and Rodrigues-like rotation formula.
    
    Mathematical background:
    - The rotation is constructed by building an orthonormal frame F using vectors a and b
    - A 2D rotation matrix G is constructed in the plane spanned by a and b
    - The final rotation R = F @ G @ F^(-1) transforms a towards b
    
    Args:
        A: torch.Tensor of shape (3,) - source vector to rotate
        b: torch.Tensor of shape (3,) or None - target direction vector
           If None, defaults to [0, 1, 0] (y-axis)
    
    Returns:
        r: torch.Tensor of shape (3, 3) - rotation matrix that rotates a towards b
    
    Example:
        >>> a = torch.tensor([1., 0., 0.])  # x-axis
        >>> b = torch.tensor([0., 1., 0.])  # y-axis
        >>> R = canonical_rotation_torch(a, b)
        >>> rotated = R @ a  # Should point towards y-axis
    """
    # Default target direction is y-axis if not specified
    if b is None:
        b = torch.tensor([0., 1., 0.], dtype=A.dtype, device=A.device)
    
    # Handle degenerate case: zero vector cannot be rotated
    if torch.norm(A) == 0:
        return torch.eye(3, dtype=A.dtype, device=A.device)
    
    # Normalize source vector
    a = A / torch.norm(A)
    
    # Handle case where b is already aligned with a (or opposite)
    # Check if the component of b perpendicular to a is zero
    b_perp = b - torch.dot(a, b) * a  # Projection of b perpendicular to a
    if torch.norm(b_perp) == 0:
        return torch.eye(3, dtype=a.dtype, device=a.device)

    # Compute rotation parameters
    ab_dot = torch.dot(a, b)              # cos(theta) where theta is angle between a and b
    ab_cross_norm = torch.norm(torch.cross(a, b))  # sin(theta) * |a| * |b|
    
    # Construct 2D rotation matrix G in the plane spanned by a and b
    # This is essentially a Givens rotation in 2D
    # G rotates the first basis vector towards the projection of b onto the (a, b_perp) plane
    g = torch.tensor([[ab_dot, -ab_cross_norm, 0.],
                      [ab_cross_norm, ab_dot, 0.],
                      [0., 0., 1.]], dtype=a.dtype, device=a.device)

    # Construct orthonormal frame F using Gram-Schmidt process
    # F[:, 0] = normalized a (first basis vector)
    # F[:, 1] = normalized component of b perpendicular to a (second basis vector)
    # F[:, 2] = cross product (third basis vector, perpendicular to both)
    f = torch.zeros((3, 3), dtype=a.dtype, device=a.device)
    f[:, 0] = a
    f[:, 1] = (b - torch.dot(a, b) * a) / torch.norm(b - torch.dot(a, b) * a)
    f[:, 2] = torch.cross(b, a)

    # Apply similarity transformation: R = F @ G @ F^(-1)
    # This rotates from standard basis -> F basis -> apply 2D rotation -> back to standard basis
    # The result is a rotation in 3D that aligns a with b
    r = f @ g @ torch.linalg.inv(f)

    return r  

def qvec2rotmat(qvec):
    return torch.tensor([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
    K = torch.tensor([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = torch.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], torch.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec