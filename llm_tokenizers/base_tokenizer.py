"""
Base tokenizer wrapper module.

This module provides a general interface for working with any tokenizer
from Hugging Face transformers library.
"""

from typing import List, Dict, Optional, Union
import torch
from transformers import AutoTokenizer


class BaseTokenizerWrapper:
    """
    A general wrapper class for any Hugging Face tokenizer.
    
    This class works with ANY model (BERT, GPT-2, T5, etc.) because
    AutoTokenizer automatically detects the correct tokenizer type.
    
    Attributes:
        tokenizer: The underlying Hugging Face tokenizer
        model_name: Name of the model used for tokenization
        vocabulary: Dictionary mapping tokens to token IDs
        reverse_vocab: Dictionary mapping token IDs to tokens
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the tokenizer.
        
        Args:
            model_name: Name of the model to use.
                       Examples:
                       - "t5-small", "t5-base", "t5-large" (T5 models)
                       - "bert-base-uncased" (BERT models)
                       - "gpt2" (GPT-2 models)
                       - "distilbert-base-uncased" (DistilBERT models)
                       - Any model from Hugging Face Hub
        """
        self.model_name = model_name
        # AutoTokenizer automatically detects the correct tokenizer type!
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._vocabulary = None
        self._reverse_vocab = None
    
    @property
    def vocabulary(self) -> Dict[str, int]:
        """Get the vocabulary mapping (tokens -> token IDs)."""
        if self._vocabulary is None:
            self._vocabulary = self.tokenizer.get_vocab()
        return self._vocabulary
    
    @property
    def reverse_vocab(self) -> Dict[int, str]:
        """Get the reverse vocabulary mapping (token IDs -> tokens)."""
        if self._reverse_vocab is None:
            self._reverse_vocab = {v: k for k, v in self.vocabulary.items()}
        return self._reverse_vocab
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.vocabulary)
    
    def encode(
        self,
        text: Union[str, List[str]],
        padding: bool = False,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = False
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text (string or list of strings)
            padding: Whether to pad sequences to the same length
            return_tensors: Return type ("pt" for PyTorch tensors, None for lists)
            max_length: Maximum length of the sequence
            truncation: Whether to truncate sequences longer than max_length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' keys
        """
        return self.tokenizer(
            text,
            padding=padding,
            return_tensors=return_tensors,
            max_length=max_length,
            truncation=truncation
        )
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = False
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs or tensor
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = False
    ) -> List[str]:
        """
        Decode a batch of token IDs to text.
        
        Args:
            token_ids: List of token ID lists or tensor
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            List of decoded text strings
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of token strings
        """
        return self.tokenizer.tokenize(text)
    
    def get_special_tokens(self) -> Dict[str, Optional[Union[str, int]]]:
        """
        Get special tokens and their IDs.
        
        Note: Different models have different special tokens.
        Some tokens may be None if not used by the model.
        
        Returns:
            Dictionary with special token information
        """
        return {
            'pad_token': self.tokenizer.pad_token,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token': self.tokenizer.eos_token,
            'eos_token_id': self.tokenizer.eos_token_id,
            'bos_token': self.tokenizer.bos_token,
            'bos_token_id': self.tokenizer.bos_token_id,
            'unk_token': self.tokenizer.unk_token,
            'unk_token_id': self.tokenizer.unk_token_id,
            'sep_token': self.tokenizer.sep_token,
            'sep_token_id': self.tokenizer.sep_token_id,
            'cls_token': self.tokenizer.cls_token,
            'cls_token_id': self.tokenizer.cls_token_id,
            'mask_token': self.tokenizer.mask_token,
            'mask_token_id': self.tokenizer.mask_token_id,
        }
    
    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = False,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = False
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        Make the tokenizer callable (convenience method).
        
        Args:
            text: Input text (string or list of strings)
            padding: Whether to pad sequences
            return_tensors: Return type ("pt" for PyTorch tensors)
            max_length: Maximum length of the sequence
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' keys
        """
        return self.encode(text, padding, return_tensors, max_length, truncation)

