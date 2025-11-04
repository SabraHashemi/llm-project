"""
Token Embeddings Visualization - PCA and Cosine Similarity

This script demonstrates how to visualize token embeddings using PCA
and cosine similarity, based on solution-01-t5.ipynb.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_tokenizers import BaseTokenizerWrapper
from llm_models import Seq2SeqModelLoader


def visualize_token_embeddings():
    """Visualize token embeddings using PCA and cosine similarity."""
    
    print("=" * 70)
    print("Token Embeddings Visualization")
    print("=" * 70)
    print("\n📝 EXPLANATION:")
    print("   This demonstrates how the model represents words as vectors.")
    print("   - Each word gets converted to a high-dimensional vector (embedding)")
    print("   - PCA reduces these to 2D for visualization")
    print("   - Cosine similarity shows how 'similar' words are to each other")
    print("   - Words that are semantically related should be close together!\n")
    
    # Load tokenizer and model
    print("🔄 Loading tokenizer and model...")
    tokenizer = BaseTokenizerWrapper("t5-small")
    model = Seq2SeqModelLoader("t5-small")
    
    # Words to visualize
    words = [
        "chair",
        "table",
        "plate",
        "knife",
        "spoon",
        "horse",
        "goat",
        "sheep",
        "cat",
        "dog",
    ]
    
    print(f"\n📝 Words to visualize:")
    for i, word in enumerate(words, 1):
        print(f"   {i}. {word}")
    
    # Tokenize words and get embeddings
    print(f"\n🔢 STEP 1: Getting token embeddings")
    print(f"   Tokenizing words and extracting embeddings from model.shared...")
    
    word_tokens = tokenizer.encode(words, return_tensors="pt", padding=True)["input_ids"][:, 0]
    print(f"   Token IDs: {word_tokens.tolist()}")
    
    # Get embeddings from the shared embedding layer
    with torch.no_grad():
        token_embeddings = model.model.shared(word_tokens).cpu().detach().numpy()
    
    print(f"   Embedding shape: {token_embeddings.shape}")
    print(f"   → Each word is represented as a {token_embeddings.shape[1]}-dimensional vector")
    
    # PCA Visualization
    print(f"\n📊 STEP 2: PCA Visualization (2D projection)")
    print(f"   Reducing {token_embeddings.shape[1]} dimensions to 2D for visualization...")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(token_embeddings)
    
    print(f"   Explained variance: {pca.explained_variance_ratio_}")
    print(f"   Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    print(f"   → This shows how much information is preserved in 2D")
    
    # Create PCA plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=100, alpha=0.6)
    
    # Add labels
    for i, word in enumerate(words):
        ax.text(X_pca[i, 0] + 0.5, X_pca[i, 1] + 0.5, word, fontsize=10)
    
    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title('Token Embeddings - PCA Visualization\n(Words close together are semantically similar)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    output_file = project_root / "examples" / "token_embeddings_pca.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n   ✅ PCA plot saved to: {output_file}")
    plt.close()
    
    # Cosine Similarity
    print(f"\n📊 STEP 3: Cosine Similarity Matrix")
    print(f"   Computing how similar each word is to every other word...")
    print(f"   (Values range from -1 to 1, where 1 = identical, 0 = unrelated)")
    
    similarity_matrix = cosine_similarity(token_embeddings)
    
    print(f"\n   Similarity matrix shape: {similarity_matrix.shape}")
    print(f"   → Shows pairwise similarity between all {len(words)} words")
    
    # Create cosine similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(similarity_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label='Cosine Similarity')
    
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_yticklabels(words)
    ax.set_title('Cosine Similarity Matrix\n(Red = similar, Blue = different)', fontsize=14)
    
    # Add similarity values to the plot
    for i in range(len(words)):
        for j in range(len(words)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    output_file = project_root / "examples" / "cosine_similarity_matrix.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Cosine similarity plot saved to: {output_file}")
    plt.close()
    
    # Print some interesting similarities
    print(f"\n💡 INTERESTING OBSERVATIONS:")
    print(f"   Most similar pairs:")
    
    # Find top similar pairs (excluding diagonal)
    similarities = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            similarities.append((words[i], words[j], similarity_matrix[i, j]))
    
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    for i, (word1, word2, sim) in enumerate(similarities[:5], 1):
        print(f"   {i}. '{word1}' ↔ '{word2}': {sim:.3f}")
    
    print(f"\n   Least similar pairs:")
    for i, (word1, word2, sim) in enumerate(similarities[-5:], 1):
        print(f"   {i}. '{word1}' ↔ '{word2}': {sim:.3f}")
    
    print(f"\n✅ Visualization complete!")
    print(f"   Check the 'examples' folder for the generated plots:")
    print(f"   - token_embeddings_pca.png")
    print(f"   - cosine_similarity_matrix.png")
    print(f"\n   📝 Interpretation:")
    print(f"   - Words that are close in the PCA plot are semantically similar")
    print(f"   - High cosine similarity (>0.5) means words have similar meanings")
    print(f"   - Notice how 'cat' and 'dog' are close (both animals)")
    print(f"   - Notice how 'chair' and 'table' are close (both furniture)")
    print(f"   - Notice how 'knife' and 'spoon' are close (both utensils)")


if __name__ == "__main__":
    try:
        visualize_token_embeddings()
    except ImportError as e:
        print(f"❌ Error: Missing required package!")
        print(f"   {e}")
        print("\n   Please install dependencies:")
        print("   pip install scikit-learn matplotlib")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

