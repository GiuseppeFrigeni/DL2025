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
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCODLoss(nn.Module):
    def __init__(self, num_train_samples, num_classes, embedding_dim, device, initial_u_std=1e-9):
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim # Dimension of GNN output Z_B (before final MLP)
        self.device = device

        # Learnable parameter u for each training sample
        # NCOD initializes u from N(1e-8, 1e-9). GCOD paper mentions 0.
        # Let's use small random init as in NCOD.
        self.u_all_samples = nn.Parameter(torch.normal(mean=1e-8, std=initial_u_std, size=(num_train_samples,), device=device),requires_grad=True)
        
        self.current_class_centroids = torch.zeros(num_classes, embedding_dim, device=device)
        self.cross_entropy_noreduction = nn.CrossEntropyLoss(reduction='none')

    def get_u_for_batch(self, indices):
        return self.u_all_samples[indices.to(self.device).long()]

    @torch.no_grad() # Centroid calculation should not involve gradients for model parameters
    def update_class_centroids(self, embeddings_epoch, labels_epoch, u_values_epoch, epoch, total_epochs):
        """
        Update class centroids based on embeddings of cleaner samples.
        embeddings_epoch: All embeddings from the training set for this epoch.
        labels_epoch: All labels from the training set.
        u_values_epoch: All u values for the training set.
        """
        self.current_class_centroids.fill_(0.0)
        counts = torch.zeros(self.num_classes, device=self.device)

        for c in range(self.num_classes):
            class_mask = (labels_epoch == c)
            if class_mask.sum() == 0:
                # Fallback: random initialization or use previous centroid if no samples for this class
                self.current_class_centroids[c] = torch.randn(self.embedding_dim, device=self.device) * 0.01
                continue

            class_embeddings = embeddings_epoch[class_mask]
            class_u_values = u_values_epoch[class_mask]

            # NCOD: "gradually decreased the number of c-labeled samples ...
            # Samples are selected from the subset with the lowest values of u."
            # "only 50% of these samples are employed in the final epoch"
            
            # Determine number of samples to use for centroid calculation
            # Starts with all, reduces to 50%
            percentage_to_use = 1.0 - 0.5 * (epoch / total_epochs)
            num_samples_in_class = class_embeddings.size(0)
            num_to_select = max(1, int(num_samples_in_class * percentage_to_use))

            if num_samples_in_class > 0:
                sorted_indices = torch.argsort(class_u_values)
                selected_indices = sorted_indices[:num_to_select]
                
                if len(selected_indices) > 0:
                    centroid_c = class_embeddings[selected_indices].mean(dim=0)
                    self.current_class_centroids[c] = centroid_c
                    counts[c] = len(selected_indices)
                else: # Should not happen if num_to_select >=1
                    self.current_class_centroids[c] = torch.randn(self.embedding_dim, device=self.device) * 0.01

        # Normalize centroids
        norms = torch.norm(self.current_class_centroids, p=2, dim=1, keepdim=True)
        # Add epsilon to norms to avoid division by zero for zero vectors
        self.current_class_centroids = self.current_class_centroids / (norms + 1e-8)


    def calculate_soft_labels_y_bar(self, embeddings_batch, targets_batch_one_hot):
        """
        Calculate soft labels y_bar for the batch based on NCOD.
        embeddings_batch: Latent representations Z_B for the current batch.
        targets_batch_one_hot: One-hot encoded dataset labels for the batch.
        """
        # Normalize batch embeddings
        h_i = F.normalize(embeddings_batch, p=2, dim=1) # Normalized sample embeddings

        # Get corresponding class centroids (h_c_i in NCOD)
        # targets_batch_one_hot is [batch_size, num_classes]
        # self.current_class_centroids is [num_classes, embedding_dim]
        # We need h_c_i for each sample i, where c_i is its class.
        class_indices = torch.argmax(targets_batch_one_hot, dim=1) # Get class indices [0, C-1]
        h_c_for_batch = self.current_class_centroids[class_indices] # [batch_size, embedding_dim]

        # Similarity: h_i^T * h_c_i
        similarity = (h_i * h_c_for_batch).sum(dim=1) # Cosine similarity, shape [batch_size]
        
        # y_bar_i = [max(similarity_i, 0)] * y_i (one-hot target)
        y_bar_magnitudes = torch.clamp(similarity, min=0.0) # Shape [batch_size]
        y_bar_batch = targets_batch_one_hot * y_bar_magnitudes.unsqueeze(1) # Shape [batch_size, num_classes]
        
        return y_bar_batch


    def forward(self, logits, embeddings_batch, targets_batch, u_batch, training_accuracy_epoch):
        batch_size = logits.size(0)
        targets_one_hot = F.one_hot(targets_batch.long(), num_classes=self.num_classes).float().to(self.device)

        # Calculate y_bar_B using NCOD's method (soft labels)
        y_bar_batch = self.calculate_soft_labels_y_bar(embeddings_batch, targets_one_hot)

        # L1 (GCOD Eq. 4)
        # L_CE(fo(Z_B) + atrain * diag(u_B) * y_B_hard, y_bar_B_soft)
        # y_B_hard is the one-hot original/dataset label
        diag_u_y_hard = torch.diag_embed(u_batch) @ targets_one_hot # [B,B] @ [B,C] -> [B,C]
        modified_logits_L1 = logits + training_accuracy_epoch * diag_u_y_hard
        
        # CE loss expects target indices for y_bar_B if it's not one-hot probabilities
        # If y_bar_batch is [B, C] of probabilities, CE needs to be handled carefully.
        # Standard CE(logits, class_indices) or CE(log_softmax(logits), prob_targets)
        # The NCOD paper uses y_bar (ỹ in their notation) as the target for LCE.
        # LCE( f(θ,xi)+ui*ỹi, ỹi ). This means ỹi must be a probability distribution.
        # Our y_bar_batch is currently scaled one-hot. To make it a distribution for CE:
        # If y_bar_batch is to be used as target probabilities, it should sum to 1.
        # NCOD's L1: LCE(f(θ,xi)+ui*ỹi, ỹi). This ỹi is the soft label.
        # The GCOD paper's L1 uses y_bar_B (soft) as target for CE.

        # If y_bar_batch represents target probabilities (might need normalization if not already)
        # loss1_per_sample = - (y_bar_batch * F.log_softmax(modified_logits_L1, dim=1)).sum(dim=1)
        # For L_CE(logits, target_indices) where target is y_bar's "class":
        # This doesn't quite fit if y_bar is soft.
        # Let's assume the standard interpretation of CE with soft targets:
        loss1_per_sample = -torch.sum(y_bar_batch * F.log_softmax(modified_logits_L1, dim=1), dim=1)
        L1 = loss1_per_sample.mean()


        # L2 (GCOD Eq. 6)
        # 1/|C| * || y_hat_B + diag(u_B) * y_B_hard - y_B_hard ||^2
        pred_probs = F.softmax(logits, dim=1)
        y_hat_batch_one_hot = F.one_hot(torch.argmax(pred_probs, dim=1), num_classes=self.num_classes).float().to(self.device)
        
        # diag(u_B)y_B_hard term
        u_y_hard_term = u_batch.unsqueeze(1) * targets_one_hot

        L2_term_inside_norm = y_hat_batch_one_hot + u_y_hard_term - targets_one_hot
        L2 = (1.0 / self.num_classes) * L2_term_inside_norm.pow(2).sum(dim=1).mean()


        # L3 (GCOD Eq. 8) - this remains specific to GCOD and might need fine-tuning
        log_probs_model = F.log_softmax(logits, dim=1)
        u_batch_clamped = torch.clamp(u_batch, 1e-7, 1.0 - 1e-7)
        target_kl_L3_scalar = torch.sigmoid(-torch.log(u_batch_clamped)) # [batch_size]
        
        # Create a target distribution for KL divergence for L3
        # The paper's DKL{L, sigma(-log(u))} where L=log(diag(fo(ZB)yB)) (log prob of true class)
        # This is still ambiguous. A common way to use KL for regularization with u:
        # Penalize deviation from uniform for noisy samples (large u)
        # Or encourage confidence for clean samples (small u)
        # The GCOD paper aims to "regulate the alignment of model predictions with the true class for clean samples
        # while preventing alignment for noisy samples (large u)".
        # This implies:
        # For clean samples (small u, so target_kl_L3 is large, close to 1 via sigmoid): DKL(model_prob_true_class || large_value)
        # For noisy samples (large u, so target_kl_L3 is small, close to 0.5 via sigmoid): DKL(model_prob_true_class || small_value)

        # Let P_model_true_class be softmax(logits).gather(1, targets_batch.long().unsqueeze(1))
        # Let Q_target be target_kl_L3_scalar
        # This still doesn't form a D_KL between distributions easily.

        # Re-interpreting L3: (1 - atrain) * DKL( sigma(-log(u_B)) || P(y_B | Z_B) )
        # where P(y_B | Z_B) is model's predicted prob for the true class y_B.
        # And sigma(-log(u_B)) acts as a target confidence for the true class.
        # This would be a KL between two Bernoulli distributions (target_kl_L3_scalar vs model_prob_true_class) for each sample.
        
        model_prob_true_class = F.softmax(logits, dim=1).gather(1, targets_batch.long().unsqueeze(1)).squeeze(1)
        model_prob_true_class_clamped = torch.clamp(model_prob_true_class, 1e-7, 1.0 - 1e-7)
        
        kl_term_1 = target_kl_L3_scalar * torch.log(target_kl_L3_scalar / model_prob_true_class_clamped)
        kl_term_2 = (1 - target_kl_L3_scalar) * torch.log((1 - target_kl_L3_scalar) / (1 - model_prob_true_class_clamped) + 1e-7) # Add eps for log
        
        L3_per_sample = (1.0 - training_accuracy_epoch) * (kl_term_1 + kl_term_2)
        L3 = L3_per_sample.mean()
        if torch.isnan(L3) or torch.isinf(L3): # Add check for L3
            print("Warning: L3 is NaN or Inf. Clamping u or model_prob might need adjustment.")
            L3 = torch.tensor(0.0, device=self.device, requires_grad=True)


        loss_model = L1 + L3
        loss_u = L2
        return loss_model, loss_u