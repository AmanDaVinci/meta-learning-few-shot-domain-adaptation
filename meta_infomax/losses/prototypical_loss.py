import torch
import torch.nn as nn


class PrototypicalLoss(nn.Module):
    ''' Loss as euclidean distance between prototypes and queries '''

    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x_embeds: torch.Tensor, n_support: int, n_query: int, n_domains: int=1):
        ''' Vectorized implementation of prototypical loss given ordered input embeddings

        The function does not need to be provided with the actual target labels.
        This information comes implicit with the arrangement of the embedding batch.
        An embedding batch is a 2D matrix where the 1st dimension contains:
        text samples from both negative and positive sentiment of n_domains 
        ordered as n_support items of label 0 from 1st domain followd by the n_query items
        then followed by n_support items of label 1 from 1st domain followed by the n_query items
        and the same ordering follows for every domain in the n_domains.
        The 2nd dimension contains the embeddings of each sample.
        This ordering allows for an efficient vectorized computation of prototypical loss.

        Since we have two labels for any domain, the number of classes will be twice the n_domains.
        The class_idxs is used to extract these support set elements and query set elements
        and group them according to the domain and label they belong to.
        The dist_matrix holds the euclidean distance of every query from all the class prototypes.
        The target_idxs is designed to compute the prototypical loss as follows:
        compute the log softmax of query distance from k-class across all classes 
        aggregate the log softmax values only for the actual class positions.
        The same principle can be used to compute accuracy: for every query,
        find the maximum negative log probability (minimum distance) across all the classes

        Parameters
        ---
        x_embeds:
        text embeddings of shape: N x D, where N = n_classes * (n_support + n_query)

        n_support:
        size of the support set 

        n_query:
        size of the support set 

        n_domains:
        number of domains in the episode

        Return
        ---
        results: tuple(torch.FloatTensor)

        loss:
        sum of negative log probability of each query's euclidean distance from the class prototype

        accuracy:
        accuracy between predicted class of query and actual class 
        '''
        n_classes = n_domains * 2
        x_embeds = x_embeds.to('cpu')

        class_idxs = torch.arange(len(x_embeds)).view(n_classes, n_support+n_query)
        support_idxs = class_idxs[:, :n_support]
        query_idxs = class_idxs[:, n_support:]

        prototypes = x_embeds[support_idxs].mean(1)                # (n_classes x D)
        queries = x_embeds[query_idxs.flatten()]                   # (n_classes*n_query x D)
        dist_matrix = self._euclidean_dist(queries, prototypes)    # (n_classes*n_query) x (n_classes)

        log_p_y = self.log_softmax(-dist_matrix)
        log_p_y = log_p_y.view(n_classes, n_query, n_classes)

        target_idxs = torch.arange(n_classes, dtype=torch.long).view(n_classes, 1, 1)
        target_idxs = target_idxs.expand(n_classes, n_query, 1)
        loss = -log_p_y.gather(dim=2, index=target_idxs).squeeze().view(-1).mean()

        _, y_pred = log_p_y.max(dim=2)        
        accuracy = (y_pred == target_idxs.squeeze()).float().mean()

        return loss, accuracy

    def _euclidean_dist(self, x, y):
        ''' Compute euclidean distance between a (N x D) & (M x D) tensor '''
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)