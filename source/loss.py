import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional # For type hinting

class SCELoss(nn.Module):
    def __init__(self, alpha: float, beta: float, num_classes: int, epsilon: float = 1e-7, reduction: str = 'mean'):
        super(SCELoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")

        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction

        # We will calculate per-sample CE loss and then apply weights if provided
        self.cross_entropy_unweighted = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Model predictions (raw scores) before softmax.
                                   Shape: (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (class indices).
                                    Shape: (batch_size,).
            class_weights (Optional[torch.Tensor]): A manual rescaling weight given to each class.
                                                    If given, has to be a Tensor of size C (num_classes).
                                                    Shape: (num_classes,).
        Returns:
            torch.Tensor: The calculated SCE loss.
        """
        # --- 0. Determine sample_weights if class_weights are provided ---
        sample_weights: Optional[torch.Tensor] = None
        if class_weights is not None:
            if class_weights.ndim != 1 or class_weights.size(0) != self.num_classes:
                raise ValueError(f"class_weights must be a 1D tensor of size num_classes ({self.num_classes}), "
                                 f"but got shape {class_weights.shape}")
            # Get the weight for each sample based on its target class
            # Ensure class_weights is on the same device as targets
            sample_weights = class_weights[targets.long()].to(logits.device) # Shape: (batch_size,)

        # --- 1. Calculate Cross Entropy (CE) ---
        # Get unweighted per-sample CE loss
        ce_loss_per_sample = self.cross_entropy_unweighted(logits, targets) # Shape: (batch_size,)

        # --- 2. Calculate Reverse Cross Entropy (RCE) ---
        pred_probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        rce_loss_per_sample = -torch.sum(pred_probs * torch.log(targets_one_hot + self.epsilon), dim=1) # Shape: (batch_size,)

        # --- 3. Combine CE and RCE for per-sample loss ---
        # Note: Here we are weighting the individual CE and RCE terms before combining.
        # An alternative is to combine them first, then apply sample_weights to the combined loss.
        # Applying to individual terms is common.
        
        weighted_ce_loss_per_sample = ce_loss_per_sample
        weighted_rce_loss_per_sample = rce_loss_per_sample

        if sample_weights is not None:
            weighted_ce_loss_per_sample = ce_loss_per_sample * sample_weights
            weighted_rce_loss_per_sample = rce_loss_per_sample * sample_weights
            
        # Combine potentially weighted CE and RCE terms
        loss_per_sample = self.alpha * weighted_ce_loss_per_sample + self.beta * weighted_rce_loss_per_sample

        # --- 4. Apply reduction ---
        if self.reduction == 'mean':
            if sample_weights is not None:
                # For a true weighted average of the combined loss:
                # (loss_per_sample defined above already has weights multiplied in)
                # return loss_per_sample.sum() / sample_weights.sum() if sample_weights.sum() > 0 else loss_per_sample.mean()
                # However, if alpha*CE*w + beta*RCE*w is desired, then sum(alpha*CE*w + beta*RCE*w) / N (or / sum(w))
                # The current 'loss_per_sample' is already (alpha*CE*w + beta*RCE*w).
                # So, simple mean of this is sum(alpha*CE*w + beta*RCE*w) / N.
                # This effectively means that weights modulate the magnitude of loss for samples from different classes.
                loss = torch.mean(loss_per_sample)
            else:
                loss = torch.mean(loss_per_sample)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_per_sample) # sum of (alpha*CE*w + beta*RCE*w)
        elif self.reduction == 'none':
            loss = loss_per_sample
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return loss