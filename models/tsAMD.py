import torch
import torch.nn as nn

from models.common import RevIN
from models.common import DDI
from models.common import MDM
from models.tsmoe import AMS


class AMD(nn.Module):
    """Implementation of AMD."""

    def __init__(self, input_shape, pred_len, n_block, dropout, patch, k, c, alpha, target_slice, norm=True):
        super(AMD, self).__init__()

        self.target_slice = target_slice
        self.norm = norm

        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])

        self.pastmixing = MDM(input_shape, k=k, c=c, layernorm=False)

        self.fc_blocks = nn.ModuleList([DDI(input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=False)
                                        for _ in range(n_block)])

        self.moe = AMS(input_shape, pred_len, ff_dim=2048, dropout=dropout, num_experts=8, top_k=2)

    def forward(self, x):
        # [batch_size, seq_len, feature_num]

        # layer norm
        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]

        # [batch_size, seq_len, feature_num]
        x = torch.transpose(x, 1, 2)
        # [batch_size, feature_num, seq_len]

        time_embedding = self.pastmixing(x)

        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # MOE
        x, moe_loss = self.moe(x, time_embedding)  # seq_len -> pred_len

        # [batch_size, feature_num, pred_len]
        x = torch.transpose(x, 1, 2)
        # [batch_size, pred_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'denorm', self.target_slice)
        # [batch_size, pred_len, feature_num]

        if self.target_slice:
            x = x[:, :, self.target_slice]

        return x, moe_loss
