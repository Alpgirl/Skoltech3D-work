import torch


class OpenCVCameraModel(torch.nn.Module):
    def __init__(self, focal, principal, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4):
        super().__init__()
        # Radial
        self.k1 = torch.nn.Parameter(torch.tensor(k1), requires_grad=False)
        self.k2 = torch.nn.Parameter(torch.tensor(k2), requires_grad=False)
        self.k3 = torch.nn.Parameter(torch.tensor(k3), requires_grad=False)
        self.k4 = torch.nn.Parameter(torch.tensor(k4), requires_grad=False)
        self.k5 = torch.nn.Parameter(torch.tensor(k5), requires_grad=False)
        self.k6 = torch.nn.Parameter(torch.tensor(k6), requires_grad=False)

        # Tangential
        self.p1 = torch.nn.Parameter(torch.tensor(p1), requires_grad=False)
        self.p2 = torch.nn.Parameter(torch.tensor(p2), requires_grad=False)

        # Prism
        self.s1 = torch.nn.Parameter(torch.tensor(s1), requires_grad=False)
        self.s2 = torch.nn.Parameter(torch.tensor(s2), requires_grad=False)
        self.s3 = torch.nn.Parameter(torch.tensor(s3), requires_grad=False)
        self.s4 = torch.nn.Parameter(torch.tensor(s4), requires_grad=False)

        # Pinhole
        self.focal = torch.nn.Parameter(torch.tensor(focal), requires_grad=False)
        self.principal = torch.nn.Parameter(torch.tensor(principal), requires_grad=False)

    def project(self, xyz):
        r"""Finds image coordinates of the point projections.
        The origin of the image coordinates is at the top-left corner of the top-left pixel.

        Parameters
        ----------
        xyz : torch.Tensor
            of shape [3, points_n]

        Returns
        -------
        uv : torch.Tensor
            of shape [2, points_n]
        """
        return project(xyz, self.focal, self.principal, self.k1, self.k2, self.k3, self.k4, self.k5, self.k6,
                       self.p1, self.p2, self.s1, self.s2, self.s3, self.s4)

    @property
    def pinhole_intrinsics(self):
        intrinsics = torch.zeros(3, 3)
        intrinsics[2, 2] = 1
        intrinsics[0, 0] = self.focal[0].item()
        intrinsics[1, 1] = self.focal[1].item()
        intrinsics[0, 2] = self.principal[0].item()
        intrinsics[1, 2] = self.principal[1].item()
        return intrinsics


def project(xyz, focal, principal, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4):
    r"""Finds image coordinates of the point projections.
    The origin of the image coordinates is at the top-left corner of the top-left pixel.

    Parameters
    ----------
    xyz : torch.Tensor
        of shape [3, points_n]

    Returns
    -------
    uv : torch.Tensor
        of shape [2, points_n]
    """
    # Normalize
    uv = xyz[:2] / xyz[2:3]

    # Distort
    u, v = uv
    r2 = u.pow(2).add_(v.pow(2))
    r4 = r2 * r2
    r6 = r4 * r2

    radial = r6.mul_(k3);
    del r6
    radial = radial.add_(r4 * k2).add_(r2 * k1).add_(1)
    full = uv * radial;
    del radial

    prism = r4 * torch.stack([s2, s4]).view(2, 1);
    del r4
    prism = prism.add_(r2 * torch.stack([s1, s3]).view(2, 1))
    full = full.add_(prism);
    del prism

    tangential = u * v * (torch.stack([p1, p2]).view(2, 1) * 2);
    del u, v
    _ = uv.pow_(2).mul_(2).add_(r2);
    del r2, uv
    tangential = tangential.mul_(torch.stack([p2, p1]).view(2, 1))
    full = full.add_(tangential);
    del tangential

    uv = full;
    del full

    # Apply pinhole
    uv = uv.mul_(focal.unsqueeze(1)).add_(principal.unsqueeze(1))

    return uv
