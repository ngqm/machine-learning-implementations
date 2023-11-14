import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,N,k]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to >initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        
        # Conv1D with kernel size 1 is equivalent to Linear numerically but 
        # faster for sharing weights between multiple points in the batch 
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - global_feature: [B,1024]
            - local_feature: [B,64,N]
            - input_transform: [B,3,3]
            - feature_transform: [B,64,64]
        """


        # transpose to [B,3,N] for dimensional compatibility
        pointcloud = pointcloud.transpose(2, 1)

        # input transform

        if self.input_transform: 
            pointcloud_tnet_input = self.stn3(pointcloud)  # [B,3,3]
            pointcloud = pointcloud.transpose(2, 1)  # [B,N,3]
            pointcloud = torch.bmm(pointcloud, pointcloud_tnet_input)
            pointcloud = pointcloud.transpose(2, 1)  # [B,3,N]
        
        # first mlp
        pointcloud = self.mlp1(pointcloud)  # [B,64,N]

        # feature transform
        if self.feature_transform:
            pointcloud_tnet_feature = self.stn64(pointcloud)  # [B,64,64]
            pointcloud = pointcloud.transpose(2, 1)  # [B,N,64]
            local_feature = torch.bmm(pointcloud, pointcloud_tnet_feature)
            local_feature = local_feature.transpose(2, 1)  # [B,64,N]
            # second mlp
            pointcloud = self.mlp2(local_feature)  # [B,1024,N]
        else:
            # second mlp
            local_feature = pointcloud
            pointcloud = self.mlp2(pointcloud)  # [B,1024,N]

        # max-pooling
        global_feature = torch.max(pointcloud, 2)[0]  # [B,1024]

        if self.feature_transform and self.input_transform:
            return global_feature, local_feature, pointcloud_tnet_input, pointcloud_tnet_feature
        elif self.feature_transform:
            return global_feature, local_feature, None, pointcloud_tnet_feature
        elif self.input_transform:
            return global_feature, local_feature, pointcloud_tnet_input, None
        else:
            return global_feature, local_feature, None, None


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.Dropout(0.3), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, num_classes), nn.LogSoftmax(dim=1)
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - input_transform [B,3,3]
            - feature_transform [B,64,64]
        """
        global_feature, _, input_transform, feature_transform = self.pointnet_feat(pointcloud)
        logits = self.mlp(global_feature)
        return logits, input_transform, feature_transform


class PointNetPartSeg(nn.Module):
    def __init__(self):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        self.num_parts = 50
        self.pointnet_feat = PointNetFeat(input_transform=True, feature_transform=True)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, self.num_parts, 1), nn.LogSoftmax(dim=1)
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - input_transform: [B,3,3]
            - feature_transform: [B,64,64]
        """
        
        # get global and local features
        global_feature, local_feature, input_transform, feature_transform = self.pointnet_feat(pointcloud)
        # concatenate global and local features
        # local_feature is [B, 64, N] and global_feature is [B, 1024]
        combined_feature = torch.cat([local_feature, global_feature.view(-1, 1024, 1).repeat(1, 1, local_feature.shape[-1])], dim=1)
        # first mlp
        point_feature = self.mlp1(combined_feature)  # [B,128,N]
        # second mlp
        logits = self.mlp2(point_feature)  # [B,50,N]
        return logits, input_transform, feature_transform



class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat(input_transform=True, feature_transform=True)

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        self.decoder = nn.Sequential(
            nn.Linear(1024, num_points//4), nn.BatchNorm1d(num_points//4), nn.ReLU(),
            nn.Linear(num_points//4, num_points//2), nn.BatchNorm1d(num_points//2), nn.ReLU(),
            nn.Linear(num_points//2, num_points), nn.Dropout(0.3), nn.BatchNorm1d(num_points), nn.ReLU(),
            nn.Linear(num_points, num_points*3)
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        global_feature, _, _, _ = self.pointnet_feat(pointcloud)
        pointcloud = self.decoder(global_feature)
        # reshape to [B,N,3]
        pointcloud = pointcloud.reshape(pointcloud.shape[0], -1, 3)
        return pointcloud


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()