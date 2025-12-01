"""
Hierarchical Decoder for Mobile-Hi-SAM
Generates 3-level hierarchy: paragraph → line → word masks
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class HierarchicalDecoder(nn.Module):
    """
    Three-level hierarchical decoder that generates:
    - Paragraph masks (coarse level)
    - Line masks (medium level)
    - Word masks (fine level)
    
    Based on Hi-SAM's HiDecoder but adapted for MobileSAM features.
    """
    
    def __init__(
        self,
        transformer_dim: int = 256,
        transformer=None,
        num_multimask_outputs: int = 3,
        activation=nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        
        # IoU token for each hierarchy level
        self.iou_token_para = nn.Embedding(1, transformer_dim)
        self.iou_token_line = nn.Embedding(1, transformer_dim)
        self.iou_token_word = nn.Embedding(1, transformer_dim)
        
        # Mask tokens for each level
        self.mask_tokens_para = nn.Embedding(num_multimask_outputs, transformer_dim)
        self.mask_tokens_line = nn.Embedding(num_multimask_outputs, transformer_dim)
        self.mask_tokens_word = nn.Embedding(num_multimask_outputs, transformer_dim)
        
        # Output upscaling for each level
        self.output_upscaling_para = nn.Sequential(
        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
             nn.BatchNorm2d(transformer_dim // 4),  # ✅ FIXED
             activation(),
             nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
             activation(),
        )

        self.output_upscaling_line = nn.Sequential(
             nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
             nn.BatchNorm2d(transformer_dim // 4),  # ✅ FIXED
             activation(),
             nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
             activation(),
        )
        self.output_upscaling_word = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(transformer_dim // 4),  # ✅ FIXED
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        # Hypernetworks for dynamic mask prediction
        self.output_hypernetworks_mlps_para = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
        )
        self.output_hypernetworks_mlps_line = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
        )
        self.output_hypernetworks_mlps_word = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
        )
        
        # IoU prediction heads
        self.iou_prediction_head_para = MLP(
            transformer_dim, iou_head_hidden_dim, num_multimask_outputs, iou_head_depth
        )
        self.iou_prediction_head_line = MLP(
            transformer_dim, iou_head_hidden_dim, num_multimask_outputs, iou_head_depth
        )
        self.iou_prediction_head_word = MLP(
            transformer_dim, iou_head_hidden_dim, num_multimask_outputs, iou_head_depth
        )
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for all three hierarchy levels.
        
        Args:
            image_embeddings: (B, C, H, W) - Image features from encoder
            image_pe: (1, C, H, W) - Positional encoding
            sparse_prompt_embeddings: (B, N, C) - Point/box prompts
            dense_prompt_embeddings: (B, C, H, W) - Dense prompts (optional)
            multimask_output: Whether to return multiple masks per level
            
        Returns:
            para_masks: (B, 1 or 3, H*4, W*4) - Paragraph masks
            para_iou: (B, 1 or 3) - Paragraph IoU predictions
            line_masks: (B, 1 or 3, H*4, W*4) - Line masks
            line_iou: (B, 1 or 3) - Line IoU predictions
            word_masks: (B, 1 or 3, H*4, W*4) - Word masks
            word_iou: (B, 1 or 3) - Word IoU predictions
        """
        
        # Get masks and IoU for each level
        para_masks, para_iou = self._predict_level(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            self.mask_tokens_para,
            self.iou_token_para,
            self.output_upscaling_para,
            self.output_hypernetworks_mlps_para,
            self.iou_prediction_head_para,
            multimask_output,
        )
        
        line_masks, line_iou = self._predict_level(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            self.mask_tokens_line,
            self.iou_token_line,
            self.output_upscaling_line,
            self.output_hypernetworks_mlps_line,
            self.iou_prediction_head_line,
            multimask_output,
        )
        
        word_masks, word_iou = self._predict_level(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            self.mask_tokens_word,
            self.iou_token_word,
            self.output_upscaling_word,
            self.output_hypernetworks_mlps_word,
            self.iou_prediction_head_word,
            multimask_output,
        )
        
        return para_masks, para_iou, line_masks, line_iou, word_masks, word_iou
    
    def _predict_level(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor],
        mask_tokens: nn.Embedding,
        iou_token: nn.Embedding,
        output_upscaling: nn.Module,
        output_hypernetwork: nn.Module,
        iou_head: nn.Module,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict masks for a single hierarchy level"""
        
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            mask_tokens=mask_tokens,
            iou_token=iou_token,
            output_upscaling=output_upscaling,
            output_hypernetwork=output_hypernetwork,
            iou_head=iou_head,
        )
        
        # Select best mask if single output
        if multimask_output:
            mask_slice = slice(0, self.num_multimask_outputs)
        else:
            mask_slice = slice(0, 1)
        
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        
        return masks, iou_pred
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor],
        mask_tokens: nn.Embedding,
        iou_token: nn.Embedding,
        output_upscaling: nn.Module,
        output_hypernetwork: nn.Module,
        iou_head: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core mask prediction logic"""
        
        # Concatenate output tokens
        output_tokens = torch.cat([iou_token.weight, mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        
        # Expand per-image data in batch direction
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings if dense_prompt_embeddings is not None else src
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        
        # Run transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_multimask_outputs), :]
        
        # Upscale mask embeddings
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = output_upscaling(src)
        
        # Generate masks
        hyper_in_list = []
        for i in range(self.num_multimask_outputs):
            hyper_in_list.append(output_hypernetwork(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        # Generate IoU predictions
        iou_pred = iou_head(iou_token_out)
        
        return masks, iou_pred


class MLP(nn.Module):
    """Simple MLP with LayerNorm"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = nn.functional.sigmoid(x)
        return x
