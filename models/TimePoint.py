from functools import partial

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

optional = {
    "encoder_dims": [64, 64, 128, 128],
    "encoder_type": "wtconv",
    "descriptor_dim": 256,
    "stride": 2,
}


def args_update(parser):
    parser.add_argument(
        "--encoder_dims",
        type=int,
        nargs="+",
        default=None,
        help="Dimensions of the encoder layers.",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default=None,
        choices=["dense", "wtconv"],
        help="Type of encoder to use: 'dense' for dense convolutional layers, 'wtconv' for wavelet transform convolutional layers.",
    )
    parser.add_argument(
        "--descriptor_dim",
        type=int,
        default=None,
        help="Dimension of the descriptor output.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for the convolutional layers in the encoder.",
    )


class TimePoint(nn.Module):
    """
    Paper: https://arxiv.org/pdf/2505.23475
    Source: https://github.com/BGU-CS-VIL/TimePoint/blob/main/TimePoint/models/timepoint.py
    """

    def __init__(self, configs):
        super().__init__()

        if configs.encoder_type == "dense":
            self.encoder = Encoder1D(configs.input_channels, configs.encoder_dims)
        elif configs.encoder_type == "wtconv":
            self.encoder = WTConvEncoder1D(
                configs.input_channels, configs.encoder_dims, stride=configs.stride
            )

        encoder_output_channels = configs.encoder_dims[-1]
        self.detector_head = DetectorHead1D(encoder_output_channels, cell_size=8)
        self.descriptor_head = DescriptorHead1D(
            encoder_output_channels, configs.descriptor_dim
        )

        # Compute parameters for each component
        encoder_params = count_parameters(self.encoder)
        detector_params = count_parameters(self.detector_head)
        descriptor_params = count_parameters(self.descriptor_head)
        total_params = encoder_params + detector_params + descriptor_params
        # Print the results
        print(f"Total number of trainable parameters: {total_params}")
        print(f"Encoder parameters: {encoder_params}")
        print(f"Detector head parameters: {detector_params}")
        print(f"Descriptor head parameters: {descriptor_params}")

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Input x: [N, C, L]
        N, C, L = x.shape
        # Extract features
        features = self.encoder(x)
        # Keypoint detection map [N, 1, L]
        detection_proba = self.detector_head(features)
        # Descriptors [N, descriptor_dim, L]
        descriptors = self.descriptor_head(features)
        # upscale might pad to multiples of 8
        detection_proba = detection_proba[:, :, :L]
        descriptors = descriptors[:, :, :L]
        # return detection_proba, descriptors
        return detection_proba.permute(0, 2, 1)

    def get_topk_points(self, x, kp_percent=1, nms_window=5):
        """
        Extract descriptors and keypoints from input signal.
        Args:
            x [N, C=1, L]: input batch of univariate time series
            kp_percent = percentage of the series length (L) to keep. if kp_percent>=1, returns the entire series.


        Returns:

        detection: Keypoint detection map [N, L] - topK keypoints, the rest are zero out
        descriptors: Descriptors [N, descriptor_dim, L]
        sorted_topk_indices: a list of len num_kp (L*kp_percent) of keypoint's timestep in their original order.
            for instace: full_timesteps: [0, 1, 2, 3, 4, 5]
                         detection_proba: [0.1, 0.3, 0, 0, 0.9, 0]
                         if num_kp = 3, then:
                         sorted_topk_indices = [0, 1, 4]

        """
        N, C, L = x.shape
        # Extract features
        features = self.encoder(x)
        # Keypoint detection map [N, 1, L]
        detection_proba = self.detector_head(features)[:, :, :L]
        descriptors = self.descriptor_head(features)[:, :, :L]
        # Non-maximum suppression (input is N,L, squeeze channel dim)
        detection_proba = detection_proba.squeeze(1)
        detection_proba = non_maximum_suppression(
            detection_proba, window_size=nms_window
        )
        # get top k points
        if kp_percent < 1:
            num_kp = int(kp_percent * L)
            sorted_topk_indices, detection_proba = get_topk_in_original_order(
                descriptors, detection_proba, K=num_kp
            )
        else:
            sorted_topk_indices = torch.arange(L)
        return sorted_topk_indices, detection_proba, descriptors


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Encoder1D(nn.Module):
    """
    Encoder that downsamples the input signal by a factor of 8.
    """

    def __init__(self, input_channels=1, dims=[64, 64, 128, 128]):
        super(Encoder1D, self).__init__()
        self.layer1 = ConvBlock1D(input_channels, dims[0], stride=1)
        self.layer2 = ConvBlock1D(dims[0], dims[1], stride=2)
        self.layer3 = ConvBlock1D(dims[1], dims[2], stride=2)
        self.layer4 = ConvBlock1D(dims[2], dims[3], stride=2)

    def forward(self, x):
        # Input x: [N, C, L]
        x = self.layer1(x)  # [N, base_channels, L]
        x = self.layer2(x)  # [N, base_channels, L/2]
        x = self.layer3(x)  # [N, base_channels*2, L/4]
        x = self.layer4(x)  # [N, base_channels*2, L/8]
        return x  # Feature map of size L/8


class WTConvEncoder1D(nn.Module):
    """
    Encoder that downsamples the input signal by a factor of 8.
    """

    def __init__(
        self, input_channels=1, dims=[64, 64, 128, 128], stride=2, wt_levels=[3, 3, 3]
    ):
        super(WTConvEncoder1D, self).__init__()
        self.stride = stride
        self.layer1 = ConvBlock1D(input_channels, dims[0], stride=1, padding="same")
        self.layer2 = WTConvBlock1D(
            dims[0], dims[1], stride=self.stride, wt_levels=wt_levels[0]
        )  # stride=2 to downsample
        self.layer3 = WTConvBlock1D(
            dims[1], dims[2], stride=self.stride, wt_levels=wt_levels[1]
        )
        self.layer4 = WTConvBlock1D(
            dims[2], dims[3], stride=self.stride, wt_levels=wt_levels[2]
        )

    def forward(self, x):
        # Input x: [N, C, L]
        x = self.layer1(x)  # [N, base_channels, L]
        x = self.layer2(x)  # [N, base_channels, L/2]
        x = self.layer3(x)  # [N, base_channels*2, L/4]
        x = self.layer4(x)  # [N, base_channels*2, L/8]
        return x  # Feature map of size L/8


class DetectorHead1D(nn.Module):
    """
    Detector Head for predicting keypoint probability map.
    """

    def __init__(self, input_channels, cell_size=8):
        super(DetectorHead1D, self).__init__()
        self.cell_size = cell_size
        self.conv = nn.Conv1d(
            input_channels, cell_size + 1, kernel_size=1
        )  # Output channels: cell_size + 1 (dustbin)

    def forward(self, x):
        # x: [N, C, L/8]
        N, C, Lc = x.shape  # Lc = L/8
        x = self.conv(x)  # [N, cell_size + 1, Lc]

        # Reshape to [N, cell_size + 1, Lc]
        # Softmax over the cell_size + 1 channels (including dustbin)
        x = F.sigmoid(x)
        # Remove dustbin (last channel)
        x = x[:, :-1, :]  # [N, cell_size, Lc]

        # Reshape to [N, 1, L]
        x = x.permute(0, 2, 1).reshape(N, 1, Lc * self.cell_size)

        return x  # Keypoint probability map of size [N, 1, L]


class DescriptorHead1D(nn.Module):
    """
    Descriptor Head for generating feature descriptors.
    """

    def __init__(self, input_channels, descriptor_dim=256):
        super(DescriptorHead1D, self).__init__()
        self.conv = nn.Conv1d(input_channels, descriptor_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode="linear", align_corners=False)

    def forward(self, x):
        # x: [N, C, L/8]
        x = self.conv(x)  # [N, descriptor_dim, L/8]
        x = self.upsample(x)  # [N, descriptor_dim, L]
        # x = F.normalize(x, p=2, dim=1)  # L2 norm along channel dimension, now performed at loss.
        return x


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, l = x.shape
    pad = filters.shape[2] // 2 - 1
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, l // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, l_half = x.shape
    pad = filters.shape[2] // 2 - 1
    x = x.reshape(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x


class WTConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type="db1",
    ):
        super(WTConv1d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )
        self.base_scale = _ScaleModule([1, in_channels, 1])

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels * 2,
                    in_channels * 2,
                    kernel_size,
                    padding="same",
                    stride=1,
                    dilation=1,
                    groups=in_channels * 2,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [
                _ScaleModule([1, in_channels * 2, 1], init_scale=0.1)
                for _ in range(self.wt_levels)
            ]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(
                torch.ones(in_channels, 1, 1), requires_grad=False
            )

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if curr_shape[2] % 2 > 0:
                curr_pads = (0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 2, shape_x[3])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:2, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, : curr_shape[2]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.stride > 1:
            x = F.conv1d(
                x,
                self.stride_filter,
                bias=None,
                stride=self.stride,
                groups=self.in_channels,
            )

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        stride=1,
        norm=nn.BatchNorm1d,
        act=nn.ReLU,
        padding=1,
    ):
        super(ConvBlock1D, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            norm(c_out),
            act(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class WTConvBlock1D(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        stride=1,
        norm=nn.BatchNorm1d,
        act=nn.ReLU,
        wt_levels=3,
    ):
        super(WTConvBlock1D, self).__init__()
        self.layer = nn.Sequential(
            WTConv1d(
                c_in, c_in, kernel_size=kernel_size, wt_levels=wt_levels, stride=stride
            ),
            nn.Conv1d(c_in, c_out, kernel_size=1, stride=1, padding=0),
            norm(c_out),
            act(),
        )

    def forward(self, x):
        return self.layer(x)


def get_topk_in_original_order(X_desc, X_probas, K):
    """
    Get the descriptors of the top K keypoints from X_keypoints without changing their original order.

    Args:
        X_desc (torch.Tensor): The descriptors associated with keypoints, shape [N, C, L].
        X_keypoints (torch.Tensor): Tensor of keypoint probabilities, shape [N, L].
        K (int): Number of top elements to select per sample.

    Returns:
        X_topk (torch.Tensor): Tensor containing the descriptors of the top K keypoints per sample, shape [N, C, K].
    """
    N, C, L = X_desc.shape
    assert X_probas.shape == (N, L), "X_keypoints must have shape (N, L)"

    device = X_probas.device
    if K >= L:
        return X_probas, X_desc

    # Get the indices of the top K values per sample
    topk_values, topk_indices = torch.topk(X_probas, K, dim=1)
    # topk_indices: shape [N, K]

    # Sort the indices per sample to maintain original order
    sorted_topk_indices, _ = torch.sort(topk_indices, dim=1)
    # sorted_topk_indices: shape [N, K]

    # Expand indices for gathering
    indices_expanded = sorted_topk_indices.unsqueeze(1).expand(
        -1, C, -1
    )  # Shape: [N, C, K]

    # Gather descriptors along the L dimension (time steps)
    X_topk = torch.gather(X_desc, dim=2, index=indices_expanded)  # Shape: [N, C, K]

    return sorted_topk_indices, X_topk


def non_maximum_suppression(detection_prob, window_size=7):
    """
    Apply non-maximum suppression to the detection map.

    Args:
        detection_map: Tensor of shape [N, L].
        window_size: Size of the window for NMS.
        threshold: Detection threshold.

    Returns:
        keypoints: Tensor of shape [N, L], boolean mask of keypoints after NMS.
    """
    # NMS
    if isinstance(detection_prob, np.ndarray):
        detection_prob = torch.from_numpy(detection_prob)
    # prepare input
    N, L = detection_prob.shape
    # (1, L' < L)
    pooled, pooled_idx = F.max_pool1d(
        detection_prob,
        kernel_size=window_size,
        stride=window_size,
        padding=window_size // 2,
        return_indices=True,
    )

    # Squeeze dim=1 from proba, make our life easier if only one sample
    if len(pooled.shape) == 3:
        detection_prob = detection_prob.squeeze()
        pooled_idx = pooled_idx.squeeze()
    # pooled_idx array of ints, turn to bool
    zero_out = torch.ones_like(detection_prob)
    for i in range(N):
        zero_out[i, pooled_idx[i]] = 0
    # zero out everything but max pooled
    detection_prob[zero_out.type(torch.bool)] = 0
    return detection_prob
