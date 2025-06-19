from .utils import *


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss).__init__()
    def NT_XENT(self, embeddings1, embeddings2, temperature = 1):
        batch_size = embeddings1.shape[0]
        
        features = torch.cat([embeddings1, embeddings2], dim=0)
        similarity = F.cosine_similarity(features[None, :, :], features[:, None, :], dim=-1) / temperature
        similarity_np = similarity.detach().numpy()
        
        identity = torch.eye(2 * batch_size, dtype=torch.int16)
        self_mask = 1 - identity
        
        numerator = torch.exp(similarity)
        denominator = torch.sum(torch.exp(similarity * self_mask), dim=1)

        loss_matrix = -torch.log(numerator / denominator)
        
        loss = torch.tensor(0, dtype=torch.float32)
        for i in range(0, batch_size):
            loss += loss_matrix[i, i + batch_size] + loss_matrix[i + batch_size, i]
        loss /= 2 * batch_size
        return similarity_np, loss
    def infoNCE(self, embeddings1, embeddings2, temperature = 1):
        batch_size = embeddings1.shape[0]
        
        # concatenate features to get pairwise cosine similarities
        features = torch.concat([embeddings1, embeddings2], dim=0)
        
        # cosine similarity
        similarity = F.cosine_similarity(features.unsqueeze(0), features.unsqueeze(1), dim=-1) / temperature
        similarity_np = similarity.detach().numpy()
        
        
        # e^sim
        similarity_exp = torch.exp(similarity)
        
        self_similarity_exp = similarity_exp[:batch_size, :batch_size]
        aug_similarity_exp = similarity_exp[:batch_size, batch_size:2*batch_size].diag()
        
        
        inv_mask = torch.ones((batch_size, batch_size), dtype=torch.bool)
        inv_mask.fill_diagonal_(0)
        
        self_similarity_exp = self_similarity_exp[inv_mask]
        self_similarity_exp = self_similarity_exp.view((batch_size, batch_size - 1))
        
        
        
        numerator = aug_similarity_exp
        denominator = torch.sum(self_similarity_exp, dim=1)
        
        term = torch.div(numerator, denominator)
        loss = -torch.log(term)
        loss = torch.mean(loss)
            
        # loss = torch.tensor(0, dtype=torch.float32, device=device)
        # for i in torch.arange(0, batch_size):
        #     inv_mask = torch.ones(batch_size, dtype=torch.bool)
        #     inv_mask[i] = 0
        #     # similarity_exp[i, i+batch_size])
        #     # print(torch.sum(similarity_exp[i][:batch_size][inv_mask]))
        #     term = similarity_exp[i, i+batch_size] / torch.sum(similarity_exp[i][:batch_size][inv_mask])
        #     loss += -torch.log(term) 
        # loss /= batch_size
        # mask_np = mask.detach().numpy()
        # print(pd.DataFrame(mask_np))
        return similarity_np, loss
        

class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR).__init__()
    def __call__(self, embeddings1, embeddings2, temperature = 1):
        CL = ContrastiveLoss()
        return CL.infoNCE(embeddings1, embeddings2, temperature)

    
    
# class BYOL(nn.Module):
    

class SupCon(nn.Module):
    """
    Supervised Contrastive Learning Loss: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.1):
        super(SupCon, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [batch_size * n_views, feature_dim]
            labels: ground truth of shape [batch_size]
            mask: contrastive mask of shape [batch_size, batch_size], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            # For when we provide batch_size * n_views features
            batch_size = features.shape[0] // 2  # Assuming 2 views
            features = features.view(batch_size, 2, -1)
        else:
            batch_size = features.shape[0]
            
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # Number of views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [batch_size * n_views, feature_dim]
        
        # Apply L2 normalization to features
        contrast_feature = F.normalize(contrast_feature, dim=1)
        
        # Compute logits
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        # Compute similarity matrix (dot product of normalized vectors = cosine similarity)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss



class MultiClassNPairLoss(nn.Module):
    def __init__(self):
        super(MultiClassNPairLoss).__init__()
    def __call__(self, batch1, batch2):
        N = len(batch1)
        outer_term = torch.tensor(0, dtype=torch.float16)
        for i in range(N):
            inner_term = torch.tensor(0, dtype=torch.float16)
            fi = batch1[i]
            fi_plus = batch2[i]

            fi = fi / torch.linalg.vector_norm(fi)
            fi_plus = fi_plus / torch.linalg.vector_norm(fi_plus)


            for j in range(N):
                if i == j:
                    continue
                fj_plus = batch2[j]
                fj_plus = fj_plus / torch.linalg.vector_norm(fj_plus)
                inner_term += torch.log(1 + torch.exp(torch.dot(fi.T, fj_plus) - torch.dot(fi.T, fi_plus)))
            outer_term += inner_term
        loss = outer_term / N
        return torch.tensor(loss, requires_grad=True)