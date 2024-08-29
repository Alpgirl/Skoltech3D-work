from skrgbd.utils import SimpleNamespace as SNS


default = SNS(
    d_min=.473,
    d_interval=.002,
    d_planes_n=256
)

configs = {
    'tis_right': default,
}
