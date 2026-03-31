
# Add project root to path
import sys

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import torch
import numpy as np

# Use package-qualified imports so they work regardless of sys.path order.
# When imported as `DataSets.data_transformation`, plain `from utils.…`
# would resolve to the top-level `utils/` package instead of `DataSets/utils/`.
from DataSets.utils.data_transformation_utils import get_masked_entry
from DataSets.utils.data_transformation_utils import get_entry_size
from DataSets.utils.data_transformation_utils import canonical_rotation
from DataSets.utils.data_transformation_utils import rotmat2qvec, qvec2rotmat, quaternion_multiply

from DataSets.utils.rotation_conversions import random_rotation

class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.01, attributes=["xyz"], mask_constant=-10):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.attributes = attributes
        self.mask_constant = mask_constant

    def __call__(self, pc):
        bsize = pc.size()[0]
        assert pc.size()[2] == get_entry_size(self.attributes)

        
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                # Duplicate random non-dropped entries and mask only
                # geodesic distance.  This avoids injecting constant
                # sentinel values that would distort downstream
                # normalization (opacity min-max, scale pc_norm, etc.).
                all_idx = np.arange(cur_pc.size()[0])
                keep_idx = np.setdiff1d(all_idx, drop_idx)
                if len(keep_idx) == 0:
                    keep_idx = all_idx  # fallback if everything dropped
                source_idx = np.random.choice(keep_idx, len(drop_idx), replace=True)
                cur_pc[drop_idx] = cur_pc[source_idx].clone()
                cur_pc[drop_idx, -1] = self.mask_constant
                pc[i, :, :] = cur_pc

        return pc


class GaussianPatchDropout(object):
    """
    Randomly drop neighbors from Gaussian patches based on geodesic distance threshold.
    
    Drops neighbors whose geodesic distance is greater than r1_min_val with a given probability.
    This augmentation helps the model learn to be robust to incomplete neighborhoods.
    
    Receives neighborhood in shape (max_neighbors, entry_size) and r1_min_val separately.
    """
    def __init__(self, max_dropout_ratio=0.5, attributes=["xyz"], mask_constant=-10):
        """
        Initialize GaussianPatchDropout.
        
        Args:
            max_dropout_ratio: Maximum ratio of neighbors to drop (0 to 1)
            attributes: List of attributes in the data (used to calculate entry size)
            mask_constant: Value to use for masking dropped entries
        """
        assert 0 <= max_dropout_ratio <= 1, "max_dropout_ratio must be in [0, 1)"
        self.max_dropout_ratio = max_dropout_ratio
        self.attributes = attributes
        self.mask_constant = mask_constant
        
        # Calculate entry size per neighbor (includes geodesic distance at end)
        self.entry_size = get_entry_size(attributes)
        
        # Get masked entry for dropping neighbors
        self.masked_entry = get_masked_entry(attributes, mask_constant)
    
    def __call__(self, neighborhood, point_features, r1_min_val, **kwargs):
        """
        Apply dropout to neighbors. If r1_min_val is provided, only dropout neighbors 
        with geodesic distance > r1_min_val. If r1_min_val is None, dropout from all neighbors.
        
        Args:
            neighborhood: tensor with shape:
                   - (max_neighbors, entry_size) for single example
                   - (bsize, max_neighbors, entry_size) for batch
            point_features: tensor with shape:
                   - (point_feature_size,) for single example
                   - (bsize, point_feature_size) for batch
            r1_min_val: scalar tensor, (bsize,) tensor of r1_min values, or None
                       If None, dropout from all valid neighbors
        
        Returns:
            Tuple of (neighborhood, point_features) with randomly dropped neighbors
        """
        # Handle both single example and batch
        is_batch = neighborhood.dim() > 2
        if not is_batch:
            neighborhood = neighborhood.unsqueeze(0)  # Add batch dimension
            point_features = point_features.unsqueeze(0)
        
        bsize = neighborhood.size()[0]
        
        for i in range(bsize):
            # Get r1_min_val for this example (None means dropout from all neighbors)
            if r1_min_val is None:
                r1_min = None
            elif torch.is_tensor(r1_min_val):
                r1_min = r1_min_val[i].item() if r1_min_val.dim() > 0 else r1_min_val.item()
            else:
                r1_min = r1_min_val
            
            # Get geodesic distances (last value in each entry)
            geodesic_distances = neighborhood[i, :, -1]
            
            # Find valid neighbors (not already masked)
            valid_mask = geodesic_distances != self.mask_constant
            
            if r1_min is None:
                # Dropout from all valid neighbors
                candidate_indices = torch.where(valid_mask)[0]
            else:
                # Find neighbors with geodesic distance > r1_min_val
                candidates_mask = (geodesic_distances > r1_min) & valid_mask
                candidate_indices = torch.where(candidates_mask)[0]
            
            if len(candidate_indices) > 0:
                # Random dropout ratio for this example
                dropout_ratio = np.random.random() * self.max_dropout_ratio
                
                # Randomly select which candidates to drop
                num_to_drop = min(int(len(candidate_indices) * dropout_ratio), 1) # Ensure at least 1 neighbor is dropped if dropout_ratio > 0
                
                if num_to_drop > 0:
                    # Randomly select indices to drop
                    drop_indices = candidate_indices[
                        torch.randperm(len(candidate_indices))[:num_to_drop]
                    ]
                    
                    # Mask the dropped neighbors by setting geodesic distance to mask_constant
                    neighborhood[i, drop_indices, -1] = self.mask_constant
        
        # Remove batch dimension if input was single example
        if not is_batch:
            neighborhood = neighborhood.squeeze(0)
            point_features = point_features.squeeze(0)
        
        return neighborhood, point_features
    
class GaussianPatchRotate(object):
    """
    Apply random rotation augmentation to Gaussian patches.
    
    Rotates XYZ coordinates and normals by a random rotation matrix.
    Receives neighborhood in shape (max_neighbors, entry_size) and point_features separately.
    
    Quaternion rotations are composed with the rotation matrix.
    """
    def __init__(self, attributes=["xyz"]):
        self.attributes = attributes
        
        # Calculate entry size per neighbor (includes geodesic distance at end)
        self.entry_size = get_entry_size(attributes)
    
    def __call__(self, neighborhood, point_features, r1_min_val=None, **kwargs):
        """
        Apply rotation to examples.
        
        Args:
            neighborhood: tensor with shape:
                   - (max_neighbors, entry_size) for single example
                   - (bsize, max_neighbors, entry_size) for batch
            point_features: tensor with shape:
                   - (point_feature_size,) for single example
                   - (bsize, point_feature_size) for batch
            r1_min_val: ignored, kept for consistent interface
        
        Returns:
            Tuple of (rotated_neighborhood, rotated_point_features)
        """
        aug_normals = kwargs.get('aug_normals')  # (max_neighbors, 3) or None

        # Handle both single example and batch
        is_batch = neighborhood.dim() > 2
        if not is_batch:
            neighborhood = neighborhood.unsqueeze(0)  # Add batch dimension
            point_features = point_features.unsqueeze(0)
            if aug_normals is not None:
                aug_normals = aug_normals.unsqueeze(0)
        
        bsize = neighborhood.size()[0]
        
        for i in range(bsize):
            # Generate random rotation 
            R = random_rotation().to(neighborhood.device)
            
            # Convert rotation matrix to quaternion for composing with existing rotations
            q_rot = rotmat2qvec(R).to(neighborhood.device)
            
            # Rotate neighborhood features
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Rotate xyz coordinates
                    neighborhood[i, :, attr_index:attr_index+3] = torch.matmul(
                        neighborhood[i, :, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "normals":
                    # Rotate normals
                    neighborhood[i, :, attr_index:attr_index+3] = torch.matmul(
                        neighborhood[i, :, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    attr_index += 3
                elif attr == "rotation":
                    # Compose quaternion rotations: q_new = q_rot * q_old
                    q_old = neighborhood[i, :, attr_index:attr_index+4]
                    q_new = quaternion_multiply(q_old, q_rot)
                    neighborhood[i, :, attr_index:attr_index+4] = q_new
                    attr_index += 4
                elif attr == "sh":
                    # SH features need rotation - but this is complex, skip for now
                    # TODO: Properly rotate spherical harmonics
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Euclidean distances don't change under rotation
                    attr_index += 1
            
            # Rotate aug_normals in-place (same R as xyz/normals)
            if aug_normals is not None:
                aug_normals[i] = torch.matmul(aug_normals[i], R.T)

            # Rotate point features (no geodesic distance here)
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Point xyz is at origin (0,0,0), rotation doesn't change it
                    attr_index += 3
                elif attr == "normals":
                    # Rotate point normals
                    point_features[i, attr_index:attr_index+3] = torch.matmul(
                        point_features[i, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    attr_index += 3
                elif attr == "rotation":
                    # Compose quaternion rotations
                    q_old = point_features[i, attr_index:attr_index+4]
                    q_new = quaternion_multiply(q_old.unsqueeze(0), q_rot).squeeze(0)
                    point_features[i, attr_index:attr_index+4] = q_new
                    attr_index += 4
                elif attr == "sh":
                    # Skip SH rotation for now
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Point to itself is 0, doesn't change
                    attr_index += 1
        
        # Remove batch dimension if input was single example
        if not is_batch:
            neighborhood = neighborhood.squeeze(0)
            point_features = point_features.squeeze(0)
        
        return neighborhood, point_features


class GaussianPatchRandomFlip(object):
    """
    Apply random flip augmentation to Gaussian patches along specified axes.
    
    Randomly flips XYZ coordinates and normals along one or more axes.
    Receives neighborhood in shape (max_neighbors, entry_size) and point_features separately.
    
    Quaternion rotations are updated to reflect the flip.
    """
    def __init__(self, attributes=["xyz"], flip_axes=[0, 1, 2], flip_prob=0.5):
        """
        Initialize GaussianPatchRandomFlip.
        
        Args:
            attributes: List of attributes in the data
            flip_axes: List of axes to potentially flip (0=x, 1=y, 2=z)
            flip_prob: Probability of flipping each axis (default: 0.5)
        """
        self.attributes = attributes
        self.flip_axes = flip_axes
        self.flip_prob = flip_prob
        
        # Calculate entry size per neighbor (includes geodesic distance at end)
        self.entry_size = get_entry_size(attributes)
    
    def __call__(self, neighborhood, point_features, r1_min_val=None, **kwargs):
        """
        Apply random flip to examples.
        
        Args:
            neighborhood: tensor with shape:
                   - (max_neighbors, entry_size) for single example
                   - (bsize, max_neighbors, entry_size) for batch
            point_features: tensor with shape:
                   - (point_feature_size,) for single example
                   - (bsize, point_feature_size) for batch
            r1_min_val: ignored, kept for consistent interface
        
        Returns:
            Tuple of (flipped_neighborhood, flipped_point_features)
        """
        aug_normals = kwargs.get('aug_normals')  # (max_neighbors, 3) or None

        # Handle both single example and batch
        is_batch = neighborhood.dim() > 2
        if not is_batch:
            neighborhood = neighborhood.unsqueeze(0)  # Add batch dimension
            point_features = point_features.unsqueeze(0)
            if aug_normals is not None:
                aug_normals = aug_normals.unsqueeze(0)
        
        bsize = neighborhood.size()[0]
        
        for i in range(bsize):
            # Determine which axes to flip for this example
            flip_mask = torch.zeros(3, dtype=torch.float32, device=neighborhood.device)
            for axis in self.flip_axes:
                if np.random.random() < self.flip_prob:
                    flip_mask[axis] = 1.0
            
            # Create flip multiplier: -1 for flipped axes, 1 for non-flipped
            flip_multiplier = 1 - 2 * flip_mask  # 1 -> -1 for flipped, 0 -> 1 for non-flipped
            
            # Skip if no axes are flipped
            if flip_mask.sum() == 0:
                continue
            
            # Flip neighborhood features
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Flip xyz coordinates
                    neighborhood[i, :, attr_index:attr_index+3] *= flip_multiplier
                    attr_index += 3
                elif attr == "normals":
                    # Flip normals
                    neighborhood[i, :, attr_index:attr_index+3] *= flip_multiplier
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    # Scale is positive, doesn't need flipping
                    attr_index += 3
                elif attr == "rotation":
                    # For quaternions, flipping affects the imaginary components
                    # If we flip axis i, we need to negate the i-th imaginary component
                    # Quaternion: (w, x, y, z) where x,y,z are imaginary
                    q = neighborhood[i, :, attr_index:attr_index+4].clone()
                    # Negate imaginary components for flipped axes
                    # A flip along axis creates a reflection, which requires adjusting quaternion
                    for axis_idx, axis in enumerate(self.flip_axes):
                        if flip_mask[axis] == 1.0:
                            # For a reflection, we need to negate corresponding quaternion component
                            q[:, axis + 1] = -q[:, axis + 1]  # +1 because q[0] is w
                    neighborhood[i, :, attr_index:attr_index+4] = q
                    attr_index += 4
                elif attr == "sh":
                    # SH features need complex handling for flips, skip for now
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Euclidean distances don't change under flip
                    attr_index += 1
            
            # Flip aug_normals in-place
            if aug_normals is not None:
                aug_normals[i] *= flip_multiplier

            # Flip point features (no geodesic distance here)
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Point xyz is at origin (0,0,0), flip doesn't change it
                    attr_index += 3
                elif attr == "normals":
                    # Flip point normals
                    point_features[i, attr_index:attr_index+3] *= flip_multiplier
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    # Scale is positive, doesn't need flipping
                    attr_index += 3
                elif attr == "rotation":
                    # Update quaternion for flip
                    q = point_features[i, attr_index:attr_index+4].clone()
                    for axis_idx, axis in enumerate(self.flip_axes):
                        if flip_mask[axis] == 1.0:
                            q[axis + 1] = -q[axis + 1]
                    point_features[i, attr_index:attr_index+4] = q
                    attr_index += 4
                elif attr == "sh":
                    # Skip SH for now
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Point to itself is 0, doesn't change
                    attr_index += 1
        
        # Remove batch dimension if input was single example
        if not is_batch:
            neighborhood = neighborhood.squeeze(0)
            point_features = point_features.squeeze(0)
        
        return neighborhood, point_features


class GaussianPatchCanonicalRotate(object):
    """
    Apply canonical rotation augmentation to Gaussian patches based on center of mass.
    
    Computes the center of mass of the neighborhood XYZ coordinates and uses canonical_rotation
    to align it with a target direction (default: y-axis). This provides a deterministic
    rotation that normalizes patch orientation.
    
    Receives neighborhood in shape (max_neighbors, entry_size) and point_features separately.
    
    Quaternion rotations are composed with the rotation matrix.
    """
    def __init__(self, attributes=["xyz"], target_direction=None, mask_constant=-10):
        self.attributes = attributes
        
        # Default target direction is y-axis
        if target_direction is None:
            self.target_direction = torch.tensor([0., 1., 0.], dtype=torch.float32)
        else:
            self.target_direction = torch.tensor(target_direction, dtype=torch.float32)
        
        # Calculate entry size per neighbor (includes geodesic distance at end)
        self.entry_size = get_entry_size(attributes)
        self.mask_constant = mask_constant
        
    
    def _compute_center_of_mass(self, xyz_coords, geodesic=None, normalize_all_neighbors=True):
        """
        Compute center of mass from XYZ coordinates.
        
        Args:
            xyz_coords: (N, 3) tensor of XYZ coordinates
            geodesic: (N,) tensor of geodesic distances (last column of neighborhood).
                Used to distinguish truly valid neighbours from dropped ones.
            normalize_all_neighbors: If True, use all non-padded entries (real_mask).
                If False, use only entries with valid geodesic (geo_valid).
            
        Returns:
            (3,) tensor representing center of mass
        """
        # real_mask: entries whose xyz is not the mask sentinel (non-padded)
        real_mask = ~torch.all(xyz_coords == self.mask_constant, dim=1)

        if normalize_all_neighbors or geodesic is None:
            valid_mask = real_mask
        else:
            # geo_valid: entries with a known geodesic distance (truly valid,
            # not dropped by dropout transforms)
            geo_valid = geodesic != self.mask_constant
            valid_mask = real_mask & geo_valid
            # Fall back to real_mask if no geo_valid entries survive
            if valid_mask.sum() == 0:
                valid_mask = real_mask
        
        if valid_mask.sum() == 0:
            # No valid points, return zero vector
            return torch.zeros(3, dtype=xyz_coords.dtype, device=xyz_coords.device)
        
        valid_coords = xyz_coords[valid_mask]
        center_of_mass = valid_coords.mean(dim=0)
        
        return center_of_mass  # Simple mean, can be weighted if needed
    
    def __call__(self, neighborhood, point_features, r1_min_val=None, **kwargs):
        """
        Apply canonical rotation to examples based on center of mass.
        
        Args:
            neighborhood: tensor with shape:
                   - (max_neighbors, entry_size) for single example
                   - (bsize, max_neighbors, entry_size) for batch
            point_features: tensor with shape:
                   - (point_feature_size,) for single example
                   - (bsize, point_feature_size) for batch
            r1_min_val: ignored, kept for consistent interface
        
        Returns:
            Tuple of (rotated_neighborhood, rotated_point_features)
        """
        aug_normals = kwargs.get('aug_normals')  # (max_neighbors, 3) or None

        # Handle both single example and batch
        is_batch = neighborhood.dim() > 2
        if not is_batch:
            neighborhood = neighborhood.unsqueeze(0)  # Add batch dimension
            point_features = point_features.unsqueeze(0)
            if aug_normals is not None:
                aug_normals = aug_normals.unsqueeze(0)
        
        bsize = neighborhood.size()[0]
        
        for i in range(bsize):
            # Find XYZ coordinates in the neighborhood
            # Assuming "xyz" is the first attribute (common case)
            xyz_start_idx = 0
            for attr_idx, attr in enumerate(self.attributes):
                if attr == "xyz":
                    break
                elif attr == "rotation":
                    xyz_start_idx += 4
                elif attr in ["normals", "scale", "sh"]:
                    xyz_start_idx += 3
                else:
                    xyz_start_idx += 1
            
            # Extract XYZ coordinates
            xyz_coords = neighborhood[i, :, xyz_start_idx:xyz_start_idx+3]
            geodesic = neighborhood[i, :, -1]  # last column = geodesic distance
            
            # Compute center of mass, respecting normalize_all_neighbors
            norm_all = kwargs.get('normalize_all_neighbors', True)
            center_of_mass = self._compute_center_of_mass(xyz_coords, geodesic=geodesic, normalize_all_neighbors=norm_all)
            
            # Compute canonical rotation matrix
            target = self.target_direction.to(neighborhood.device)
            R = canonical_rotation(center_of_mass, target)
            
            # Convert rotation matrix to quaternion for composing with existing rotations
            q_rot = rotmat2qvec(R).to(neighborhood.device)
            
            # Rotate neighborhood features
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Rotate xyz coordinates
                    neighborhood[i, :, attr_index:attr_index+3] = torch.matmul(
                        neighborhood[i, :, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "normals":
                    # Rotate normals
                    neighborhood[i, :, attr_index:attr_index+3] = torch.matmul(
                        neighborhood[i, :, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    attr_index += 3
                elif attr == "rotation":
                    # Compose quaternion rotations: q_new = q_rot * q_old
                    q_old = neighborhood[i, :, attr_index:attr_index+4]
                    q_new = quaternion_multiply(q_old, q_rot)
                    neighborhood[i, :, attr_index:attr_index+4] = q_new
                    attr_index += 4
                elif attr == "sh":
                    # SH features need rotation - but this is complex, skip for now
                    # TODO: Properly rotate spherical harmonics
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Euclidean distances don't change under rotation
                    attr_index += 1
            
            # Rotate aug_normals in-place (same R as xyz/normals)
            if aug_normals is not None:
                aug_normals[i] = torch.matmul(aug_normals[i], R.T)

            # Rotate point features (no geodesic distance here)
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Point xyz is at origin (0,0,0), rotation doesn't change it
                    attr_index += 3
                elif attr == "normals":
                    # Rotate point normals
                    point_features[i, attr_index:attr_index+3] = torch.matmul(
                        point_features[i, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    attr_index += 3
                elif attr == "rotation":
                    # Compose quaternion rotations
                    q_old = point_features[i, attr_index:attr_index+4]
                    q_new = quaternion_multiply(q_old.unsqueeze(0), q_rot).squeeze(0)
                    point_features[i, attr_index:attr_index+4] = q_new
                    attr_index += 4
                elif attr == "sh":
                    # Skip SH rotation for now
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Point to itself is 0, doesn't change
                    attr_index += 1
        
        # Remove batch dimension if input was single example
        if not is_batch:
            neighborhood = neighborhood.squeeze(0)
            point_features = point_features.squeeze(0)
        
        return neighborhood, point_features


class SparseContextDropout(object):
    """Unified dropout that replaces both GaussianPatchDropout and the old
    SparseContextDropout.

    Two complementary behaviours are applied **stochastically**:

    1. **GaussianPatchDropout mode** (probability ``p_dropout``):
       Among the *valid* neighbours whose geodesic distance exceeds
       ``r1_min_val``, randomly drop a fraction up to ``max_dropout_ratio``.
       This is equivalent to the old ``GaussianPatchDropout`` but restricted
       to valid (non-masked) neighbours only.

    2. **Sparse context mode** (probability ``1 – p_dropout``):
       Keep only *K* closest valid neighbours (K ∈ [``min_valid``,
       ``max_valid_ratio × n_valid``]) and mask the rest, simulating the
       extreme sparsity seen near the source during Fast Marching inference.

    The outer probability ``p`` gates whether *any* dropout is applied at
    all for a given example (same as before).

    Args:
        min_valid:  Minimum number of valid neighbours to keep in sparse
            mode (default 1).
        max_valid_ratio: Maximum fraction of original valid neighbours to
            keep in sparse mode (default 0.3).
        p: Probability of applying *any* dropout per example (default 0.3).
        p_dropout: Probability of choosing GaussianPatchDropout mode vs
            sparse-context mode when dropout is triggered (default 0.3).
        max_dropout_ratio: In GaussianPatchDropout mode, maximum fraction of
            eligible (geo > r1_min_val) neighbours to drop (default 0.5).
        attributes: List of attribute names.
        mask_constant: Sentinel value used for masked geodesic distance.
    """

    def __init__(
        self,
        min_valid: int = 1,
        max_valid_ratio: float = 1.00,
        p: float = 0.7,
        p_dropout: float = 0.3,
        max_dropout_ratio: float = 0.99,
        attributes=None,
        mask_constant: float = -10.0,
    ):
        if attributes is None:
            attributes = ["xyz"]
        self.min_valid = max(1, min_valid)
        self.max_valid_ratio = max_valid_ratio
        self.p = p
        self.p_dropout = p_dropout
        self.max_dropout_ratio = max_dropout_ratio
        self.attributes = attributes
        self.mask_constant = mask_constant

    def __call__(self, neighborhood, point_features, r1_min_val=None, **kwargs):
        is_batch = neighborhood.dim() > 2
        if not is_batch:
            neighborhood = neighborhood.unsqueeze(0)
            point_features = point_features.unsqueeze(0)

        bsize = neighborhood.size(0)
        for i in range(bsize):
            if np.random.random() > self.p:
                continue

            geo = neighborhood[i, :, -1]
            valid_idx = torch.where(geo != self.mask_constant)[0]
            n_valid = len(valid_idx)

            if n_valid <= self.min_valid:
                continue

            # Decide mode
            if np.random.random() < self.p_dropout:
                # --- GaussianPatchDropout mode (among valid only) ---
                if r1_min_val is not None:
                    if torch.is_tensor(r1_min_val):
                        r1_min = r1_min_val[i].item() if r1_min_val.dim() > 0 else r1_min_val.item()
                    else:
                        r1_min = r1_min_val
                    candidate_mask = geo[valid_idx] > r1_min
                    candidate_indices = valid_idx[candidate_mask]
                else:
                    candidate_indices = valid_idx

                if len(candidate_indices) > 0:
                    dropout_ratio = np.random.random() * self.max_dropout_ratio
                    num_to_drop = int(len(candidate_indices) * dropout_ratio)
                    # Ensure at least min_valid neighbours survive
                    max_droppable = n_valid - self.min_valid
                    num_to_drop = min(num_to_drop, max_droppable)
                    if num_to_drop > 0:
                        drop_indices = candidate_indices[
                            torch.randperm(len(candidate_indices))[:num_to_drop]
                        ]
                        neighborhood[i, drop_indices, -1] = self.mask_constant
            else:
                # --- Sparse context mode (keep K closest valid) ---
                max_keep = max(self.min_valid, int(n_valid * self.max_valid_ratio))
                n_keep = np.random.randint(self.min_valid, max_keep + 1)

                geo_valid = geo[valid_idx]
                keep_order = torch.argsort(geo_valid)[:n_keep]
                keep_set = set(keep_order.tolist())

                drop_local = [j for j in range(n_valid) if j not in keep_set]
                if drop_local:
                    drop_idx = valid_idx[torch.tensor(drop_local, dtype=torch.long)]
                    neighborhood[i, drop_idx, -1] = self.mask_constant

        if not is_batch:
            neighborhood = neighborhood.squeeze(0)
            point_features = point_features.squeeze(0)

        return neighborhood, point_features


class GaussianPatchSurfacePerturb(object):
    """Perturb neighbor XYZ positions along analytical surface normals.

    When ``aug_normals`` are available (passed via kwargs from the dataset),
    each valid neighbour is displaced along its surface normal by a random
    offset drawn from ``Uniform(-max_offset, +max_offset)``.  This simulates
    the off-surface displacement of Gaussians observed in real reconstructions
    while preserving the tangent-plane structure.

    If ``aug_normals`` are *not* provided, falls back to isotropic noise
    (uniform random per coordinate) for backward compatibility.

    Args:
        surface_type: Kept for config back-compat; ignored.
        max_offset: Maximum perturbation magnitude in normalized patch
            coordinates (default 0.02).
        p: Probability of applying this transform per example (default 0.5).
        max_ratio: Maximum fraction of valid neighbors to perturb per
            example (default 1.0 = all valid neighbors).  The actual
            fraction is drawn uniformly from [0, max_ratio].
        attributes: List of attribute names.
        mask_constant: Sentinel value used for masked geodesic distance.
    """

    def __init__(
        self,
        surface_type=None,
        max_offset: float = 0.02,
        p: float = 0.5,
        max_ratio: float = 1.0,
        attributes=None,
        mask_constant: float = -10.0,
    ):
        if attributes is None:
            attributes = ["xyz"]
        self.max_offset = max_offset
        self.p = p
        self.max_ratio = max_ratio
        self.attributes = attributes
        self.mask_constant = mask_constant

        # Pre-compute xyz offset within the entry
        self._xyz_offset = None
        idx = 0
        for attr in attributes:
            if attr == "xyz":
                self._xyz_offset = idx
                break
            elif attr == "rotation":
                idx += 4
            elif attr in ["normals", "scale", "sh"]:
                idx += 3
            else:
                idx += 1

    def __call__(self, neighborhood, point_features, r1_min_val=None, **kwargs):
        if self._xyz_offset is None:
            return neighborhood, point_features

        aug_normals = kwargs.get('aug_normals')  # (max_neighbors, 3) or None

        is_batch = neighborhood.dim() > 2
        if not is_batch:
            neighborhood = neighborhood.unsqueeze(0)
            point_features = point_features.unsqueeze(0)
            if aug_normals is not None:
                aug_normals = aug_normals.unsqueeze(0)

        bsize = neighborhood.size(0)
        off = self._xyz_offset

        for i in range(bsize):
            if np.random.random() > self.p:
                continue

            # Only perturb valid (non-masked) neighbors
            geo = neighborhood[i, :, -1]
            valid_mask = geo != self.mask_constant
            if valid_mask.sum() == 0:
                continue

            xyz = neighborhood[i, :, off:off + 3]  # (N, 3)

            if aug_normals is not None:
                # Directed perturbation along surface normal
                normals = aug_normals[i]  # (N, 3)
                offsets = torch.empty(xyz.size(0), 1, device=xyz.device, dtype=xyz.dtype).uniform_(
                    -self.max_offset, self.max_offset
                )
                noise = offsets * normals  # (N, 3) — displacement along normal
            else:
                # Fallback: no noise
                noise = torch.empty_like(xyz).uniform_(0, 0)

            # Zero out noise for masked neighbors
            noise[~valid_mask] = 0.0

            # Subsample: only perturb a random fraction of valid neighbors
            if self.max_ratio < 1.0:
                valid_idx = torch.where(valid_mask)[0]
                n_valid = len(valid_idx)
                ratio = np.random.random() * self.max_ratio
                n_keep = max(1, int(n_valid * ratio))
                perm = torch.randperm(n_valid, device=xyz.device)[:n_keep]
                perturb_mask = torch.zeros_like(valid_mask)
                perturb_mask[valid_idx[perm]] = True
                noise[~perturb_mask] = 0.0

            neighborhood[i, :, off:off + 3] = xyz + noise

        if not is_batch:
            neighborhood = neighborhood.squeeze(0)
            point_features = point_features.squeeze(0)

        return neighborhood, point_features


class GeodesicNoiseAugmentation(object):
    """Simulate min_input prediction error during Fast Marching inference.

    During FM propagation the model's prediction for the current point
    becomes ``min_input`` for the next wavefront.  The geodesic
    normalization subtracts ``min_input`` from every neighbour's raw
    geodesic distance, so an error ``δ`` in ``min_input`` shifts **all**
    normalized neighbour geodesics by ``-δ / scale``.

    This transform reproduces that pattern: a single additive offset is
    drawn per example and applied uniformly to every valid neighbour's
    normalised geodesic::

        offset ~ Uniform(-max_noise, +max_noise) * geo_range
        geo_noisy = geo + offset          (for all valid neighbours)

    where ``geo_range = max(valid_geos) - min(valid_geos)``.

    This is more faithful than per-neighbour multiplicative noise because
    min_input error is a shared bias, not independent per-neighbour jitter.

    Args:
        max_noise: Scale factor for the offset relative to the geodesic
            range in the patch (default 0.1 → offset up to ±10 % of range).
        p: Probability of applying the augmentation per example (default 0.5).
        max_ratio: Maximum fraction of valid neighbors whose geodesic
            values are perturbed per example (default 1.0 = all valid
            neighbors).  The actual fraction is drawn uniformly from
            [0, max_ratio].  Unperturbed neighbors keep their original
            geodesic value.
        attributes: List of attribute names (unused, kept for registry
            injection consistency).
        mask_constant: Sentinel value for masked geodesic entries.
    """

    def __init__(
        self,
        max_noise: float = 0.1,
        p: float = 0.5,
        max_ratio: float = 1.0,
        attributes=None,
        mask_constant: float = -10.0,
    ):
        if attributes is None:
            attributes = ["xyz"]
        assert 0.0 < max_noise < 1.0, "max_noise must be in (0, 1)"
        self.max_noise = max_noise
        self.p = p
        self.max_ratio = max_ratio
        self.attributes = attributes
        self.mask_constant = mask_constant

    def __call__(self, neighborhood, point_features, r1_min_val=None, **kwargs):
        is_batch = neighborhood.dim() > 2
        if not is_batch:
            neighborhood = neighborhood.unsqueeze(0)
            point_features = point_features.unsqueeze(0)

        bsize = neighborhood.size(0)
        for i in range(bsize):
            if np.random.random() > self.p:
                continue

            geo = neighborhood[i, :, -1]
            valid_mask = geo != self.mask_constant

            if valid_mask.sum() == 0:
                continue

            valid_geos = geo[valid_mask]
            geo_range = valid_geos.max() - valid_geos.min()
            if geo_range < 1e-8:
                geo_range = valid_geos.abs().max()
            if geo_range < 1e-8:
                continue

            # One offset per example — simulates shared min_input error
            offset = (torch.rand(1, device=geo.device, dtype=geo.dtype).item() * 2 - 1) * self.max_noise * geo_range

            if self.max_ratio < 1.0:
                # Subsample: only perturb a random fraction of valid neighbors
                valid_idx = torch.where(valid_mask)[0]
                n_valid = len(valid_idx)
                ratio = np.random.random() * self.max_ratio
                n_perturb = max(1, int(n_valid * ratio))
                perm = torch.randperm(n_valid, device=geo.device)[:n_perturb]
                subset_idx = valid_idx[perm]
                geo[subset_idx] = geo[subset_idx] + offset
            else:
                geo[valid_mask] = valid_geos + offset

        if not is_batch:
            neighborhood = neighborhood.squeeze(0)
            point_features = point_features.squeeze(0)

        return neighborhood, point_features

