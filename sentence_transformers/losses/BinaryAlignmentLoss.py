import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Iterable, Dict
from sentence_transformers.SentenceTransformer import SentenceTransformer

def mae_loss(input, target):
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    return torch.mean(expanded_input - expanded_target)

class BinaryAlignmentLoss(nn.Module):
    """
    BinaryAlignmentLoss is first originated from **Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN** published in NeurIPS 2018.
    Then the loss is refined in **Learning efficient binary representation for images with unsupervised deep neural networks** with additional normalization.
    Further discussion is also made in Shuffle and Learn: Minimizing Mutual Information for Unsupervised Hashing, which anneals the objective with 
    mutual information.
    
    Note: This loss is paired with output from HashLayer

    | Paper: Learning efficient binary representation for images with unsupervised deep neural networks, http://hdl.handle.net/2429/78848
    | Blog post: https://omoindrot.github.io/triplet-loss

    :param model: SentenceTransformer model
    :param consistency_loss: Consistency loss type to align the relations. Must be one of ['mse', 'mae'] 
    :param approx_penalty: Approximation penalty to control the proximity between the approximate continuous and the binary embedding

    Example::

        from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models
        from sentence_transformers.readers import InputExample

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        hash_layer = models.HashLayer(word_embedding_model.get_word_embedding_dimension(), word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, hash_layer])
        train_examples = [InputExample(texts=['Sentence from class 0'], label=0), InputExample(texts=['Another sentence from class 0'], label=0),
            InputExample(texts=['Sentence from class 1'], label=1), InputExample(texts=['Sentence from class 2'], label=2)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.BinaryAlignmentLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, reconstruct_loss='mse', approx_penalty=0.1):
        super(BinaryAlignmentLoss, self).__init__()
        assert reconstruct_loss in ["mse", "mae"]
        self.sentence_embedder = model
        if reconstruct_loss == "mse":
            self.reconstruct_loss = F.mse_loss
        elif reconstruct_loss == "mae":
            self.reconstruct_loss = mae_loss
        self.approx_penalty = approx_penalty

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        out = self.sentence_embedder(sentence_features[0])
        assert "binary_embedding_approx" in out and "binary_embedding" in out, \
            "BinaryAlignmentLoss only accept output from `sentence_transformers.models.HashLayer`."
        orig = out['sentence_embedding']
        approx = out["binary_embedding_approx"]
        bin = out["binary_embedding"]
        orig = orig.detach()
        return self.batch_binary_alignment_loss(orig, approx, bin, labels)

    def batch_binary_alignment_loss(self, x: Tensor, b: Tensor, h: Tensor, labels: Tensor):
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        dist_x = torch.matmul(x, x.t()) / torch.matmul(x_norm, x_norm.t())
        b_norm = x.norm(p=2, dim=-1, keepdim=True)
        dist_b = torch.matmul(b, b.t()) / torch.matmul(b_norm, b_norm.t())
        
        dist_x, dist_b = torch.triu(dist_x, diagonal=1), torch.triu(dist_b, diagonal=1)
        
        loss_reconstruct = self.reconstruct_loss(dist_x, dist_b)
        loss_consist = torch.mean(torch.abs(torch.pow(torch.abs(h) - 1, 3)))

        return loss_reconstruct + self.approx_penalty * loss_consist
