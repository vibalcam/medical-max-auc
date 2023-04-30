import torch
import torch.nn.functional as F


def contrastive_loss(hidden1, target, T=0.1, device=None, LARGE_NUM=1e9):
        # Get (normalized) hidden1 and hidden2.
        hidden1 = F.normalize(hidden1, p=2, dim=1)
        batch_size = hidden1.shape[0]

        ''' Note: **
        Cosine similarity matrix of all samples in batch:
        a = z_i
        b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|
        Postives:
        Diagonals of ab and ba '\'
        Negatives:
        All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''

        # mask similarities
        masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long, device=device), batch_size)
        # labels are probabilities of being similar
        labels = (target[:,None] == target[None,:]).int() * (1-masks)
        labels = labels / labels.sum(1)

        # similarity between all views (cosine since previously normalized)
        logits_aa = torch.matmul(hidden1, hidden1.T)/ T

        # mask similarities to exclude them
        logits_aa = logits_aa - masks * LARGE_NUM

        # compute cross entropy
        loss = F.cross_entropy(F.softmax(logits_aa, 1), labels)
        return loss


def contrastive_loss(hidden1, hidden2, T=0.1, device=None, LARGE_NUM=1e9):
        # if gamma != 1:
        #     raise Exception("gamma should be 1 for contrastive loss")

        # # use dynamic contrastive loss implementation instead
        # return dynamic_contrastive_loss(hidden1, hidden2, index, gamma=1.0)

        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]

        ''' Note: **
        Cosine similarity matrix of all samples in batch:
        a = z_i
        b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|
        Postives:
        Diagonals of ab and ba '\'
        Negatives:
        All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''

        # diagonal are positive labels (1), rest negative (0)
        labels = F.one_hot(torch.arange(batch_size, dtype=torch.long, device=device), batch_size * 2)
        masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long, device=device), batch_size)

        # similarity between all views (cosine since previously normalized)
        logits_aa = torch.matmul(hidden1, hidden1.T)/ T
        logits_bb = torch.matmul(hidden2, hidden2.T)/ T
        logits_ab = torch.matmul(hidden1, hidden2.T)/ T
        logits_ba = torch.matmul(hidden2, hidden1.T)/ T

        # mask diagonal (self-similarities)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = logits_bb - masks * LARGE_NUM

        # logits for a and for b
        logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1) 
        logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1) 

        # compute cross entropy
        def softmax_cross_entropy_with_logits(labels, logits):
            #logits = logits - torch.max(logits)
            expsum_neg_logits = torch.sum(torch.exp(logits), dim=1, keepdim=True)
            normalized_logits = logits - torch.log(expsum_neg_logits)
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb)
        
        # take mean of losses per batch
        loss = (loss_a + loss_b).mean()
        return loss