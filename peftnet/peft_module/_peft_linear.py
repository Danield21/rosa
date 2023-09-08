from typing import Union

import torch
import torch.nn as nn


class PeftLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: Union[int, float] = 1.0,
            bias: bool = False
    ):
        """ROSA PEFT linear layer with trainable and fixed parameters

        Args:
            in_features: number of input features
            out_features: number of output features
            rank: rank of factorized matrices
            bias: whether to include bias

        Notes:
            - Initialized with random weights and `merged` flag set to False
            - If `rank` is a float, it is interpreted as a ratio of the rank upper bound

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = self._integer_rank(rank, full_rank=min(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features, 1)) if bias else None
        self.a = nn.Parameter(torch.zeros(in_features, self.rank))
        self.b = nn.Parameter(torch.randn(self.rank, out_features))
        self.w = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.merged = False

    def initialize_weights(self, a_init: torch.Tensor = None, b_init: torch.Tensor = None, w_init: torch.Tensor = None):
        self.a.data = a_init if a_init is not None else self.a.data
        self.b.data = b_init if b_init is not None else self.b.data
        self.w.data = w_init if w_init is not None else self.w.data

    @classmethod
    def from_module(cls, linear_layer: nn.Module, rank=1.0, fan_in_fan_out=True) -> nn.Module:
        """Initialize from a nn.Linear/Conv1D module"""

        w = linear_layer.weight.data  # [out_f, in_f] or [in_f, out_f] if fan_in_fan_out
        w = w if fan_in_fan_out else w.T
        bias = linear_layer.bias.data if linear_layer.bias is not None else None

        # Initialize
        obj = cls(
            in_features=w.size(0), out_features=w.size(1), rank=rank, bias=bias is not None
        )
        a = torch.zeros(obj.in_features, obj.rank, device=w.device)
        b = torch.randn(obj.rank, obj.out_features, device=w.device)
        obj.initialize_weights(w_init=w, a_init=a, b_init=b)
        return obj

    def merge(self):
        """Merge `a` and `b` with `w` and make `w` trainable"""
        if not self.merged:
            # Merge w and ab
            self.w.data = self.a.data @ self.b.data + self.w.data

            # todo: empty a and b tensors to save memory
            # Make a, b fixed and w trainable
            self.a.requires_grad = False
            self.b.requires_grad = False
            self.w.requires_grad = True

            # Toggle merged flag
            self.merged = True
        return self

    def factorize(self):
        """Factorize `w` into `a` and `b` and make a portion of `a` and `b` trainable"""
        if not self.merged:
            self.merge()

        # Factorize
        try:
            u, s, vt = torch.linalg.svd(self.w.data, full_matrices=False)  # [in_f, r],[r,],[r, out_f]
            a = s.reshape(1, -1) * u
            b = vt
            w_hat = a @ b
        except:
            import pdb; pdb.set_trace()

        # Check reconstruction error
        assert torch.allclose(self.w.data, w_hat, atol=1e-2), "ERROR: Reconstruction error is too large"

        rank_upper_bound = min(self.w.shape)
        trainable_indices, fixed_indices = self._select_k_from_n(self.rank, rank_upper_bound, mode='random')

        # Set trainable and fixed parameters
        init_a_trainable = a[:, trainable_indices]  # [in_f, r']
        init_b_trainable = b[trainable_indices, :]  # [r', out_f]
        init_a_fixed = a[:, fixed_indices]
        init_b_fixed = b[fixed_indices, :]
        init_w = init_a_fixed @ init_b_fixed

        # Initialize
        self.a.data = init_a_trainable
        self.b.data = init_b_trainable
        self.w.data = init_w

        # Make `a`, `b` trainable and `w` fixed
        self.a.requires_grad = True
        self.b.requires_grad = True
        self.w.requires_grad = False

        # Toggle merged flag
        self.merged = False
        return self

    def __repr__(self):
        cls = self.__class__.__name__
        return (f'{cls}('
                f'rank={self.rank}, '
                f'a={self.a.shape, self.a.requires_grad}, '
                f'b={self.b.shape, self.b.requires_grad}, '
                f'w={self.w.shape, self.w.requires_grad}, '
                f'bias={(self.bias.shape, self.bias.requires_grad) if self.bias is not None else None}'
                f')')

    @staticmethod
    def _integer_rank(rank, full_rank):
        """ Convert a ratio to an integer"""
        return rank if rank > 1 else max(int(rank * full_rank), 1)

    def _select_k_from_n(self, k: int, n: int, mode: str = 'random'):
        """Choose k indices from n indices"""

        assert 0 < k <= n, "k must be an integer between 0 and n"
        assert isinstance(k, int) and isinstance(n, int), "k and n must be integers"

        if mode.lower() == 'random':
            start_idx = torch.randint(0, n - k, (1,)).item()
            end_idx = start_idx + k
            chosen_ids = torch.arange(start_idx, end_idx)
            remaining_ids = [i for i in range(n) if i not in chosen_ids]
        elif mode.lower() == 'top':
            raise NotImplementedError
        elif mode.lower() == 'bottom':
            raise NotImplementedError
        else:
            raise AttributeError(f"Unknown mode: {mode}. Mode must be one of ['random', 'top', 'bottom']")

        return chosen_ids, remaining_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x: x: [*, in_features]

        Returns:
            y: [*, out_features]
        """

        if self.merged and self.training:
            raise RuntimeError("Cannot call forward on a merged layer in training mode. ")

        if self.merged:
            # [*, in_features] @ [in_features, out_features] + [out_features, 1] = [*, out_features]
            return x @ self.w + self.bias.reshape(-1) if self.bias is not None else x @ self.w
        else:
            # [*, in_features] @ [in_features, rank] @ [rank, out_features] + [out_features, 1] = [*, out_features]
            return x @ self.a @ self.b + x @ self.w + self.bias.reshape(-1) if self.bias is not None \
                else x @ self.a @ self.b + x @ self.w