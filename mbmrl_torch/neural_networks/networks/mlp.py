from mbmrl_torch.neural_networks.networks.base import NNBase

class mlp(NNBase):
    def __init__(self, dim_in, dim_out, cuda):
        super().__init__(dim_in, dim_out, cuda)
        