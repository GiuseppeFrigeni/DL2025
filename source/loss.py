import torch
import torch.nn as nn
import torch.nn.functional as F

class SCELoss(nn.Module):
    def __init__(self, alpha: float, beta: float, num_classes: int, epsilon: float = 1e-7, reduction: str = 'mean'):
        """
        Symmetric Cross Entropy Loss.
        Combines Cross Entropy (CE) with Reverse Cross Entropy (RCE).

        Args:
            alpha (float): Weight for the CE loss term.
            beta (float): Weight for the RCE loss term.
            num_classes (int): Number of classes.
            epsilon (float): Small value to prevent log(0).
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean': the sum of the output will be divided
                             by the number of elements in the output. 'sum': the output will be summed.
                             'none': no reduction will be applied.
        """
        super(SCELoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction type: {reduction}. Must be 'mean', 'sum', or 'none'.")

        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.epsilon = epsilon  # To prevent log(0)
        self.reduction = reduction

        # Standard CrossEntropyLoss expects logits as input and class indices as target
        # It internally applies log_softmax and NLLLoss
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none') # We will handle reduction manually

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Model predictions (raw scores) before softmax.
                                   Shape: (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (class indices).
                                    Shape: (batch_size,).
        Returns:
            torch.Tensor: The calculated SCE loss.
        """
        # --- 1. Calculate Cross Entropy (CE) ---
        # nn.CrossEntropyLoss handles log_softmax internally
        ce_loss = self.cross_entropy(logits, targets) # Shape: (batch_size,)

        # --- 2. Calculate Reverse Cross Entropy (RCE) ---
        # Get model probabilities
        pred_probs = F.softmax(logits, dim=1) # Shape: (batch_size, num_classes)

        # Convert target class indices to one-hot encoding
        # Ensure targets are long type for one_hot
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float() # Shape: (batch_size, num_classes)

        # Add epsilon for numerical stability before log
        # RCE = -sum(pred_probs * log(targets_one_hot + epsilon))
        # Element-wise multiplication, then sum over classes
        rce_loss_per_sample = -torch.sum(pred_probs * torch.log(targets_one_hot + self.epsilon), dim=1) # Shape: (batch_size,)

        # --- 3. Combine CE and RCE ---
        loss_per_sample = self.alpha * ce_loss + self.beta * rce_loss_per_sample # Shape: (batch_size,)

        # --- 4. Apply reduction ---
        if self.reduction == 'mean':
            loss = torch.mean(loss_per_sample)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_per_sample)
        elif self.reduction == 'none':
            loss = loss_per_sample
        else: # Should not happen due to init check
            raise ValueError(f"Unsupported reduction: {self.reduction}")

        return loss