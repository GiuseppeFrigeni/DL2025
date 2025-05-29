import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
        sample_weights: Optional[torch.Tensor] = None
        if class_weights is not None:
            if class_weights.ndim != 1 or class_weights.size(0) != self.num_classes:
                raise ValueError(f"class_weights must be a 1D tensor of size num_classes ({self.num_classes}), "
                                 f"but got shape {class_weights.shape}")

            sample_weights = class_weights[targets.long()].to(logits.device) # Shape: (batch_size,)

        ce_loss_per_sample = self.cross_entropy_unweighted(logits, targets) # Shape: (batch_size,)

        pred_probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        rce_loss_per_sample = -torch.sum(pred_probs * torch.log(targets_one_hot + self.epsilon), dim=1) # Shape: (batch_size,)
        
        weighted_ce_loss_per_sample = ce_loss_per_sample
        weighted_rce_loss_per_sample = rce_loss_per_sample

        if sample_weights is not None:
            weighted_ce_loss_per_sample = ce_loss_per_sample * sample_weights
            weighted_rce_loss_per_sample = rce_loss_per_sample * sample_weights
            
        loss_per_sample = self.alpha * weighted_ce_loss_per_sample + self.beta * weighted_rce_loss_per_sample

        if self.reduction == 'mean':
            if sample_weights is not None:
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


class GCODLoss(nn.Module):
    def __init__(self, num_train_samples, num_classes, embedding_dim, device, initial_u_std=1e-9):
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.device = device
        self.u_all_samples = nn.Parameter(torch.normal(mean=1e-8, std=initial_u_std, size=(num_train_samples,), device=device),requires_grad=True)
        self.current_class_centroids = torch.zeros(num_classes, embedding_dim, device=device, requires_grad=False) # Centroids are not learned directly by model's optimizer
        self.cross_entropy_noreduction = nn.CrossEntropyLoss(reduction='none')

    def get_u_for_batch(self, indices):
        return self.u_all_samples[indices.to(self.device).long()]

    @torch.no_grad()
    def update_class_centroids(self, embeddings_epoch, labels_epoch, u_values_epoch, epoch, total_epochs):
        self.current_class_centroids.fill_(0.0)
        counts = torch.zeros(self.num_classes, device=self.device)
        for c in range(self.num_classes):
            class_mask = (labels_epoch == c)
            if class_mask.sum() == 0:
                self.current_class_centroids[c] = torch.randn(self.embedding_dim, device=self.device) * 0.01
                continue
            class_embeddings = embeddings_epoch[class_mask]
            class_u_values = u_values_epoch[class_mask]
            percentage_to_use = 1.0 - 0.5 * (epoch / total_epochs)
            num_samples_in_class = class_embeddings.size(0)
            num_to_select = max(1, int(num_samples_in_class * percentage_to_use))
            if num_samples_in_class > 0:
                sorted_indices = torch.argsort(class_u_values)
                selected_indices = sorted_indices[:num_to_select]
                if len(selected_indices) > 0:
                    centroid_c = class_embeddings[selected_indices].mean(dim=0)
                    self.current_class_centroids[c] = centroid_c
                else:
                    self.current_class_centroids[c] = torch.randn(self.embedding_dim, device=self.device) * 0.01
        norms = torch.norm(self.current_class_centroids, p=2, dim=1, keepdim=True)
        self.current_class_centroids = self.current_class_centroids / (norms + 1e-8)

    def calculate_soft_labels_y_bar(self, embeddings_batch, targets_batch_one_hot):
        h_i = F.normalize(embeddings_batch, p=2, dim=1)
        class_indices = torch.argmax(targets_batch_one_hot, dim=1)
        h_c_for_batch = self.current_class_centroids[class_indices]
        similarity = (h_i * h_c_for_batch).sum(dim=1)
        y_bar_magnitudes = torch.clamp(similarity, min=0.0)
        y_bar_batch = targets_batch_one_hot * y_bar_magnitudes.unsqueeze(1)
        return y_bar_batch

    def forward(self, logits, embeddings_batch, targets_batch, u_batch, training_accuracy_epoch):
        batch_size = logits.size(0)
        targets_one_hot = F.one_hot(targets_batch.long(), num_classes=self.num_classes).float().to(self.device)
        y_bar_batch = self.calculate_soft_labels_y_bar(embeddings_batch, targets_one_hot)

        diag_u_y_hard = torch.diag_embed(u_batch) @ targets_one_hot
        modified_logits_L1 = logits + training_accuracy_epoch * diag_u_y_hard
        loss1_per_sample = -torch.sum(y_bar_batch * F.log_softmax(modified_logits_L1, dim=1), dim=1)
        L1 = loss1_per_sample.mean()

        pred_probs = F.softmax(logits, dim=1)
        y_hat_batch_one_hot = F.one_hot(torch.argmax(pred_probs, dim=1), num_classes=self.num_classes).float().to(self.device)
        u_y_hard_term = u_batch.unsqueeze(1) * targets_one_hot
        L2_term_inside_norm = y_hat_batch_one_hot + u_y_hard_term - targets_one_hot
        L2 = (1.0 / self.num_classes) * L2_term_inside_norm.pow(2).sum(dim=1).mean()

        model_prob_true_class = F.softmax(logits, dim=1).gather(1, targets_batch.long().unsqueeze(1)).squeeze(1)
        model_prob_true_class_clamped = torch.clamp(model_prob_true_class, 1e-7, 1.0 - 1e-7)
        u_batch_clamped = torch.clamp(u_batch, 1e-7, 1.0 - 1e-7)
        target_kl_L3_scalar = torch.sigmoid(-torch.log(u_batch_clamped))
        
        target_kl_L3_scalar_clamped = torch.clamp(target_kl_L3_scalar, 1e-7, 1.0 - 1e-7)

        kl_term_1 = target_kl_L3_scalar_clamped * torch.log(target_kl_L3_scalar_clamped / model_prob_true_class_clamped + 1e-7) # Added epsilon for log
        kl_term_2 = (1 - target_kl_L3_scalar_clamped) * torch.log((1 - target_kl_L3_scalar_clamped) / (1 - model_prob_true_class_clamped + 1e-7) + 1e-7)

        L3_per_sample = (1.0 - training_accuracy_epoch) * (kl_term_1 + kl_term_2)
        L3 = L3_per_sample.mean()
        if torch.isnan(L3) or torch.isinf(L3):
            L3 = torch.tensor(0.0, device=self.device) # Ensure L3 is differentiable even if 0

        loss_model = L1 + L3
        loss_u = L2
        return loss_model, loss_u