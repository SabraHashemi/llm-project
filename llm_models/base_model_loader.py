"""
Base model loader module.

This module provides a general interface for loading transformer models
from Hugging Face transformers library.
"""

from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoConfig
)


class BaseModelLoader:
    """
    A general wrapper class for loading any Hugging Face transformer model.
    
    This class works with ANY model type (BERT, GPT-2, T5, etc.) by using
    the appropriate AutoModel class based on the task type.
    
    Attributes:
        model: The loaded Hugging Face model
        model_name: Name of the model
        config: Model configuration
        model_type: Type of model (seq2seq, causal, classification, etc.)
    """
    
    # Mapping of model types to their AutoModel classes
    MODEL_CLASSES = {
        'seq2seq': AutoModelForSeq2SeqLM,
        'causal': AutoModelForCausalLM,
        'classification': AutoModelForSequenceClassification,
        'masked_lm': AutoModelForMaskedLM,
        'token_classification': AutoModelForTokenClassification,
        'base': AutoModel,
    }
    
    def __init__(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ):
        """
        Initialize the model loader.
        
        Args:
            model_name: Name of the model to load (e.g., "t5-small", "bert-base-uncased")
            model_type: Type of model task. If None, will try to infer from model_name.
                       Options:
                       - 'seq2seq': For sequence-to-sequence models (T5, BART, etc.)
                       - 'causal': For causal language models (GPT-2, GPT-3, etc.)
                       - 'classification': For sequence classification (BERT for classification)
                       - 'masked_lm': For masked language models (BERT, RoBERTa)
                       - 'token_classification': For token classification (BERT for NER)
                       - 'base': Base model without task head
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional arguments passed to from_pretrained()
        """
        self.model_name = model_name
        self.model_type = model_type or self._infer_model_type(model_name)
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        
        # Load configuration
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Get the appropriate model class
        model_class = self.MODEL_CLASSES.get(self.model_type, AutoModel)
        
        # Load the model
        self.model = model_class.from_pretrained(
            model_name,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        
        # Set model to evaluation mode by default
        self.model.eval()
    
    def _infer_model_type(self, model_name: str) -> str:
        """
        Infer model type from model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Inferred model type
        """
        model_name_lower = model_name.lower()
        
        # T5 models are seq2seq
        if 't5' in model_name_lower or 'bart' in model_name_lower or 'pegasus' in model_name_lower:
            return 'seq2seq'
        
        # GPT models are causal
        if 'gpt' in model_name_lower or 'llama' in model_name_lower or 'opt' in model_name_lower:
            return 'causal'
        
        # BERT and similar are often used for classification or masked LM
        # Default to base model for flexibility
        return 'base'
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size from model config."""
        return getattr(self.config, 'vocab_size', None)
    
    @property
    def hidden_size(self) -> int:
        """Get the hidden size from model config."""
        return getattr(self.config, 'd_model', None) or getattr(self.config, 'hidden_size', None)
    
    @property
    def num_layers(self) -> int:
        """Get the number of layers from model config."""
        return getattr(self.config, 'num_layers', None) or getattr(self.config, 'num_hidden_layers', None)
    
    @property
    def num_heads(self) -> int:
        """Get the number of attention heads from model config."""
        return getattr(self.config, 'num_heads', None) or getattr(self.config, 'num_attention_heads', None)
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get model configuration information.
        
        Returns:
            Dictionary with key configuration parameters
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the model.
        
        This is a convenience method that calls the model's forward method.
        """
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """
        Make the model callable.
        
        This allows you to use the model like: model(input_ids=...)
        """
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """
        Generate text using the model.
        
        This method is available for seq2seq and causal models.
        """
        if not hasattr(self.model, 'generate'):
            raise AttributeError(f"Model type '{self.model_type}' does not support generation")
        return self.model.generate(*args, **kwargs)
    
    def to(self, device: str):
        """
        Move model to a device (CPU, CUDA, etc.).
        
        Args:
            device: Device to move model to ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
        return self


class Seq2SeqModelLoader(BaseModelLoader):
    """
    Convenience class for sequence-to-sequence models (T5, BART, etc.).
    """
    
    def __init__(self, model_name: str = "t5-small", **kwargs):
        """
        Initialize a seq2seq model loader with T5 default.
        
        Args:
            model_name: Name of the model (default: "t5-small")
            **kwargs: Additional arguments passed to BaseModelLoader
        """
        super().__init__(model_name, model_type='seq2seq', **kwargs)

