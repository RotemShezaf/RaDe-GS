import torch
import numpy as np
from utils.data_transformation_utils import get_masked_entry
from utils.data_transformation_utils import get_entry_size
from utils.data_transformation_utils import canonical_rotation
from utils.data_transformation_utils import rotmat2qvec



class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.01, attributes=["xyz"], mask_constant=-10):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.attributes = attributes
        self.mask_constant = mask_constant

    def __call__(self, pc):
        bsize = pc.size()[0]
        assert pc.size()[2] == get_entry_size(self.attributes)
        masked_entry = get_masked_entry(self.attributes, self.mask_constant)

        
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist()] = masked_entry.repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc


class GaussianPatchDropout(object):
    """
    Randomly drop neighbors from Gaussian patches based on geodesic distance threshold.
    
    Drops neighbors whose geodesic distance is greater than r1_min_val with a given probability.
    This augmentation helps the model learn to be robust to incomplete neighborhoods.
    
    Handles the flattened structure: [neighborhood_features, point_features, r1_min_val, p_u]
    where each neighbor has a geodesic distance stored at the end of its entry.
    """
    def __init__(self, max_dropout_ratio=0.5, attributes=["xyz"], max_neighbors=512, 
                 use_r1_min_val=True, mask_constant=-10):
        """
        Initialize GaussianPatchDropout.
        
        Args:
            max_dropout_ratio: Maximum ratio of neighbors to drop (0 to 1)
            attributes: List of attributes in the data (used to calculate entry size)
            max_neighbors: Maximum number of neighbors in the patch
            use_r1_min_val: Whether r1_min_val is present in the data
            mask_constant: Value to use for masking dropped entries
        """
        assert 0 <= max_dropout_ratio < 1, "max_dropout_ratio must be in [0, 1)"
        self.max_dropout_ratio = max_dropout_ratio
        self.attributes = attributes
        self.max_neighbors = max_neighbors
        self.use_r1_min_val = use_r1_min_val
        self.mask_constant = mask_constant
        
        # Calculate entry size per neighbor (includes geodesic distance at end)
        self.entry_size = get_entry_size(attributes)
        
        # Calculate point feature size (no geodesic distance in point features)
        self.point_feature_size = get_entry_size(attributes) - 1  # minus geodesic distance
        
        # Get masked entry for dropping neighbors
        self.masked_entry = get_masked_entry(attributes, mask_constant)
    
    def __call__(self, batch):
        """
        Apply dropout to neighbors with geodesic distance > r1_min_val.
        
        Args:
            batch: tensor with shape:
                   - (feature_dim,) for single example
                   - (bsize, feature_dim) for batch
                   where feature_dim includes:
                   [neighborhood (max_neighbors * entry_size), 
                    point_features (point_feature_size),
                    r1_min_val (if use_r1_min_val=True),
                    p_u]
        
        Returns:
            Batch with randomly dropped neighbors (masked with mask_constant)
        """
        # Handle both single example and batch
        is_batch = batch.dim() > 1
        if not is_batch:
            batch = batch.unsqueeze(0)  # Add batch dimension
        
        bsize = batch.size()[0]
        
        for i in range(bsize):
            # Extract r1_min_val
            if self.use_r1_min_val:
                r1_min_val = batch[i, -2].item()  # Second to last
            else:
                # If r1_min_val is not provided, skip dropout (or use alternative strategy)
                continue
            
            # Extract neighborhood features
            neighborhood_flat = batch[i, :self.max_neighbors * self.entry_size]
            neighborhood = neighborhood_flat.view(self.max_neighbors, self.entry_size)
            
            # Get geodesic distances (last value in each entry)
            geodesic_distances = neighborhood[:, -1]
            
            # Find neighbors with geodesic distance > r1_min_val
            candidates_mask = geodesic_distances > r1_min_val
            candidate_indices = torch.where(candidates_mask)[0]
            
            if len(candidate_indices) > 0:
                # Random dropout ratio for this example
                dropout_ratio = np.random.random() * self.max_dropout_ratio
                
                # Randomly select which candidates to drop
                num_to_drop = int(len(candidate_indices) * dropout_ratio)
                
                if num_to_drop > 0:
                    # Randomly select indices to drop
                    drop_indices = candidate_indices[
                        torch.randperm(len(candidate_indices))[:num_to_drop]
                    ]
                    
                    # Mask the dropped neighbors
                    #masked_entry_expanded = self.masked_entry.squeeze(0).to(batch.device)
                    neighborhood[drop_indices, -1] = self.mask_constant
                    
                    # Write back to batch
                    batch[i, :self.max_neighbors * self.entry_size] = neighborhood.flatten()
        
        # Remove batch dimension if input was single example
        if not is_batch:
            batch = batch.squeeze(0)
        
        return batch
    
class GaussianPatchRotate(object):
    """
    Apply random rotation augmentation to Gaussian patches.
    
    Rotates XYZ coordinates and normals around the y-axis by a random angle.
    Handles the flattened structure: [neighborhood_features, point_features, r1_min_val, p_u]
    where each neighbor/point has attributes in order specified by self.attributes.
    
    Quaternion rotations are composed with the rotation matrix.
    """
    def __init__(self, attributes=["xyz"], max_neighbors=512, use_r1_min_val=True):
        self.attributes = attributes
        self.max_neighbors = max_neighbors
        self.use_r1_min_val = use_r1_min_val
        
        # Calculate entry size per neighbor (includes geodesic distance at end)
        self.entry_size = get_entry_size(attributes)
        
        # Calculate point feature size (no geodesic distance in point features)
        self.point_feature_size = sum([
            4 if attr == "rotation" else 
            3 if attr in ["xyz", "normals", "scale", "sh"] else 
            1 for attr in attributes
        ])
        
    def _quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions q1 * q2.
        Quaternions are in (w, x, y, z) format.
        
        Args:
            q1: (N, 4) tensor of quaternions
            q2: (N, 4) tensor of quaternions or (4,) single quaternion
            
        Returns:
            (N, 4) tensor of quaternion products
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        
        if q2.dim() == 1:
            w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        else:
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def __call__(self, batch):
        """
        Apply rotation to examples.
        
        Args:
            batch: tensor with shape:
                   - (feature_dim,) for single example
                   - (bsize, feature_dim) for batch
                   where feature_dim includes:
                   [neighborhood (max_neighbors * entry_size), 
                    point_features (point_feature_size),
                    r1_min_val (optional),
                    p_u]
        
        Returns:
            Rotated batch with same shape
        """
        # Handle both single example and batch
        is_batch = batch.dim() > 1
        if not is_batch:
            batch = batch.unsqueeze(0)  # Add batch dimension
        
        bsize = batch.size()[0]
        
        for i in range(bsize):
            # Generate random rotation angle around y-axis
            
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            R = torch.tensor([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]], dtype=torch.float32, device=batch.device)
            
            #rotation_matrix = np.linalg.qr(np.random.randn(3, 3))[0]  # Random rotation matrix via QR decomposition
            #R = rotation_matrixr
            
            # Convert rotation matrix to quaternion for composing with existing rotations
            q_rot = rotmat2qvec( R).to(batch.device)
            
            # Extract neighborhood features
            neighborhood_flat = batch[i, :self.max_neighbors * self.entry_size]
            neighborhood = neighborhood_flat.view(self.max_neighbors, self.entry_size)
            
            # Calculate where point features start
            point_start_idx = self.max_neighbors * self.entry_size
            point_end_idx = point_start_idx + self.point_feature_size
            
            # Extract point features
            point_features = batch[i, point_start_idx:point_end_idx]
            
            # Rotate neighborhood features
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Rotate xyz coordinates
                    neighborhood[:, attr_index:attr_index+3] = torch.matmul(
                        neighborhood[:, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "normals":
                    # Rotate normals
                    neighborhood[:, attr_index:attr_index+3] = torch.matmul(
                        neighborhood[:, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    attr_index += 3
                elif attr == "rotation":
                    # Compose quaternion rotations: q_new = q_rot * q_old
                    q_old = neighborhood[:, attr_index:attr_index+4]
                    q_new = self._quaternion_multiply(q_old, q_rot)
                    neighborhood[:, attr_index:attr_index+4] = q_new
                    attr_index += 4
                elif attr == "sh":
                    # SH features need rotation - but this is complex, skip for now
                    # TODO: Properly rotate spherical harmonics
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Euclidean distances don't change under rotation
                    attr_index += 1
            
            # Geodesic distance at the end of each entry doesn't change
            # (already accounted for in entry_size)
            
            # Rotate point features (no geodesic distance here)
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Point xyz is at origin (0,0,0), rotation doesn't change it
                    attr_index += 3
                elif attr == "normals":
                    # Rotate point normals
                    point_features[attr_index:attr_index+3] = torch.matmul(
                        point_features[attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    attr_index += 3
                elif attr == "rotation":
                    # Compose quaternion rotations
                    q_old = point_features[attr_index:attr_index+4]
                    q_new = self._quaternion_multiply(q_old.unsqueeze(0), q_rot).squeeze(0)
                    point_features[attr_index:attr_index+4] = q_new
                    attr_index += 4
                elif attr == "sh":
                    # Skip SH rotation for now
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Point to itself is 0, doesn't change
                    attr_index += 1
            
            # Write back transformed data
            batch[i, :self.max_neighbors * self.entry_size] = neighborhood.flatten()
            batch[i, point_start_idx:point_end_idx] = point_features
            
            # r1_min_val and p_u at the end don't change
        
        # Remove batch dimension if input was single example
        if not is_batch:
            batch = batch.squeeze(0)
        
        return batch


class GaussianPatchCanonicalRotate(object):
    """
    Apply canonical rotation augmentation to Gaussian patches based on center of mass.
    
    Computes the center of mass of the neighborhood XYZ coordinates and uses canonical_rotation
    to align it with a target direction (default: y-axis). This provides a deterministic
    rotation that normalizes patch orientation.
    
    Handles the flattened structure: [neighborhood_features, point_features, r1_min_val, p_u]
    where each neighbor/point has attributes in order specified by self.attributes.
    
    Quaternion rotations are composed with the rotation matrix.
    """
    def __init__(self, attributes=["xyz"], max_neighbors=512, use_r1_min_val=True, target_direction=None):
        self.attributes = attributes
        self.max_neighbors = max_neighbors
        self.use_r1_min_val = use_r1_min_val
        
        # Default target direction is y-axis
        if target_direction is None:
            self.target_direction = torch.tensor([0., 1., 0.], dtype=torch.float32)
        else:
            self.target_direction = torch.tensor(target_direction, dtype=torch.float32)
        
        # Calculate entry size per neighbor (includes geodesic distance at end)
        self.entry_size = get_entry_size(attributes)
        
        # Calculate point feature size (no geodesic distance in point features)
        self.point_feature_size = sum([
            4 if attr == "rotation" else 
            3 if attr in ["xyz", "normals", "scale", "sh"] else 
            1 for attr in attributes
        ])
        
    def _quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions q1 * q2.
        Quaternions are in (w, x, y, z) format.
        
        Args:
            q1: (N, 4) tensor of quaternions
            q2: (N, 4) tensor of quaternions or (4,) single quaternion
            
        Returns:
            (N, 4) tensor of quaternion products
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        
        if q2.dim() == 1:
            w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        else:
            w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    def _compute_center_of_mass(self, xyz_coords):
        """
        Compute center of mass from XYZ coordinates.
        
        Args:
            xyz_coords: (N, 3) tensor of XYZ coordinates
            
        Returns:
            (3,) tensor representing center of mass
        """
        # Filter out masked entries (assuming mask_constant=-10 is used)
        # Valid points have non-negative or reasonable coordinates
        valid_mask = ~torch.all(xyz_coords == -10, dim=1)
        
        if valid_mask.sum() == 0:
            # No valid points, return zero vector
            return torch.zeros(3, dtype=xyz_coords.dtype, device=xyz_coords.device)
        
        valid_coords = xyz_coords[valid_mask]
        center_of_mass = valid_coords.mean(dim=0)
        
        return center_of_mass
    
    def __call__(self, batch):
        """
        Apply canonical rotation to examples based on center of mass.
        
        Args:
            batch: tensor with shape:
                   - (feature_dim,) for single example
                   - (bsize, feature_dim) for batch
                   where feature_dim includes:
                   [neighborhood (max_neighbors * entry_size), 
                    point_features (point_feature_size),
                    r1_min_val (optional),
                    p_u]
        
        Returns:
            Rotated batch with same shape
        """
        # Handle both single example and batch
        is_batch = batch.dim() > 1
        if not is_batch:
            batch = batch.unsqueeze(0)  # Add batch dimension
        
        bsize = batch.size()[0]
        
        for i in range(bsize):
            # Extract neighborhood features
            neighborhood_flat = batch[i, :self.max_neighbors * self.entry_size]
            neighborhood = neighborhood_flat.view(self.max_neighbors, self.entry_size)
            
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
            xyz_coords = neighborhood[:, xyz_start_idx:xyz_start_idx+3]
            
            # Compute center of mass
            center_of_mass = self._compute_center_of_mass(xyz_coords)
            
            # Compute canonical rotation matrix
            target = self.target_direction.to(batch.device)
            R = canonical_rotation(center_of_mass, target)
            
            # Convert rotation matrix to quaternion for composing with existing rotations
            q_rot = rotmat2qvec(R).to(batch.device)
            
            # Calculate where point features start
            point_start_idx = self.max_neighbors * self.entry_size
            point_end_idx = point_start_idx + self.point_feature_size
            
            # Extract point features
            point_features = batch[i, point_start_idx:point_end_idx]
            
            # Rotate neighborhood features
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Rotate xyz coordinates
                    neighborhood[:, attr_index:attr_index+3] = torch.matmul(
                        neighborhood[:, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "normals":
                    # Rotate normals
                    neighborhood[:, attr_index:attr_index+3] = torch.matmul(
                        neighborhood[:, attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    attr_index += 3
                elif attr == "rotation":
                    # Compose quaternion rotations: q_new = q_rot * q_old
                    q_old = neighborhood[:, attr_index:attr_index+4]
                    q_new = self._quaternion_multiply(q_old, q_rot)
                    neighborhood[:, attr_index:attr_index+4] = q_new
                    attr_index += 4
                elif attr == "sh":
                    # SH features need rotation - but this is complex, skip for now
                    # TODO: Properly rotate spherical harmonics
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Euclidean distances don't change under rotation
                    attr_index += 1
            
            # Geodesic distance at the end of each entry doesn't change
            # (already accounted for in entry_size)
            
            # Rotate point features (no geodesic distance here)
            attr_index = 0
            for attr in self.attributes:
                if attr == "xyz":
                    # Point xyz is at origin (0,0,0), rotation doesn't change it
                    attr_index += 3
                elif attr == "normals":
                    # Rotate point normals
                    point_features[attr_index:attr_index+3] = torch.matmul(
                        point_features[attr_index:attr_index+3], R.T
                    )
                    attr_index += 3
                elif attr == "opacity":
                    attr_index += 1
                elif attr == "scale":
                    attr_index += 3
                elif attr == "rotation":
                    # Compose quaternion rotations
                    q_old = point_features[attr_index:attr_index+4]
                    q_new = self._quaternion_multiply(q_old.unsqueeze(0), q_rot).squeeze(0)
                    point_features[attr_index:attr_index+4] = q_new
                    attr_index += 4
                elif attr == "sh":
                    # Skip SH rotation for now
                    attr_index += 3
                elif attr == "euclidean_distances":
                    # Point to itself is 0, doesn't change
                    attr_index += 1
            
            # Write back transformed data
            batch[i, :self.max_neighbors * self.entry_size] = neighborhood.flatten()
            batch[i, point_start_idx:point_end_idx] = point_features
            
            # r1_min_val and p_u at the end don't change
        
        # Remove batch dimension if input was single example
        if not is_batch:
            batch = batch.squeeze(0)
        
        return batch
    
    


