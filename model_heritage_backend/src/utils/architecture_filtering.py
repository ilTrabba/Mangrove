class FilteringPatterns:
    """Predefined layer exclusion patterns for distance calculations"""
    # Lista completa di pattern per layer da ESCLUDERE dal calcolo della distanza
    # utilizzando un pattern, vengono esclusi tutti i layer tranne quelli del nome del pattern

    # =============================================================================
    # 1. BASE EXCLUSIONS (Normalization, Embeddings, Heads, Pools)
    # =============================================================================
    # Queste stringhe vanno scartate SEMPRE, sia che tu voglia solo la backbone,
    # sia che tu voglia solo l'attention.
    # =============================================================================
    BASE_EXCLUSIONS = frozenset([
        # --- Normalization (copre LLaMA, GPT, BERT, ViT) ---
        'layernorm', 'layer_norm', 'ln_', '.ln.', 'ln_1', 'ln_2', 'ln_f', # GPT/ViT
        'batchnorm', 'batch_norm', '.bn.', 'bn1', 'bn2', 'bn3',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        'norm.', '_norm', '.norm', # Generic catch-all
        'norm1', 'norm2', 'norm3', # LLaMA / ViT / Swin specific
        'final_norm', 'model.norm', # LLaMA final norm

        # --- Embeddings / Input Stems ---
        'embed', 'embedding', 'embeddings',
        'token_embed', 'word_embed', 'wte', 'wpe', # GPT identifiers
        'position_embed', 'positional_embed', 'pos_embed', 'abs_pos_embed',
        'patch_embed', 'patch_embedding', # ViT
        'input_embedding', 'input_embed',
        'shared', # CRITICO: Matrice embedding condivisa in T5/BART
        'proj_stem',

        # --- Positional / Rotary / Special Tokens ---
        'rope', 'rotary', 'alibi', # Positional encodings (spesso fissi o buffer)
        'pos_bias', 'relative_position', 'rel_pos',
        'relative_attention_bias', # T5 bias
        'cls_token', 'dist_token', 'mask_token', # ViT parameters

        # --- Output Heads / Classifiers / Poolers ---
        'lm_head', 'language_model_head',
        'classifier', 'classification_head', 'cls_head',
        'segmentation_head', 'mask_head', 'det_head',
        'prediction_head', 'pred_head',
        'qa_outputs', 'seq_relationship', 'next_sentence',
        'pooler', 'global_pool', 'avgpool', 'maxpool',
        'logits', 'final_logits_bias',
        'visual_projection', 'text_projection', # CLIP specific

        # --- Final Layers Generic Names ---
        'classifier.', 'head.', 'final_fc', 'final_linear',
        'output_head', 'output_layer', 'final_layer'
    ])

    # =============================================================================
    # 2. LISTA BACKBONE ONLY
    # =============================================================================
    # Obiettivo: Tenere Attention (Q,K,V,O) + MLP (Gate, Up, Down, FC).
    # Scartiamo solo Base Exclusions.
    # =============================================================================
    BACKBONE_ONLY = BASE_EXCLUSIONS

    # =============================================================================
    # 3. LISTA ATTENTION ONLY
    # =============================================================================
    # Obiettivo: Tenere SOLO Attention (Q,K,V,O).
    # Scartiamo Base Exclusions + MLP Layers + Convolution Layers.
    # =============================================================================
    MLP_AND_CONV_EXCLUSIONS = frozenset([
        # --- Classic MLP / Feed Forward ---
        'mlp', 'ffn', 'feedforward', 'feed_forward',
        'intermediate',      # BERT: 'intermediate.dense' è la prima parte dell'MLP
        'output.dense',      # BERT: 'output.dense' è la seconda parte dell'MLP (ATTENZIONE: non scartare attention.output)
        'dense_h_to_4h', 'dense_4h_to_h', # Megatron legacy
        'fc1', 'fc2',        # Standard naming
        'linear1', 'linear2',

        # --- Modern LLM MLP (LLaMA, Mistral, Qwen) ---
        'gate_proj', # Parte del GLU (MLP)
        'up_proj',   # Parte del GLU (MLP)
        'down_proj', # Parte del GLU (MLP)

        # --- T5 / PaLM MLP ---
        'wi', 'wi_0', 'wi_1', # Weights Input (MLP)
        'wo',                 # Weights Output (MLP)

        # --- Mixture of Experts (MoE) ---
        'experts', 'block_sparse_moe',

        # --- Convolutional Layers (spesso usati come FFN in Vision) ---
        'conv', 'conv1', 'conv2', 'conv3',
        'downsample'
        ])

    # Uniamo i due set
    ATTENTION_ONLY = BASE_EXCLUSIONS | MLP_AND_CONV_EXCLUSIONS

    BACKBONE_EMBEDDING = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Head/Output layers
        'lm_head', 'language_model_head',
        'classifier', 'classification_head',
        'head', 
        'output_projection', 'output_proj',
        'score', 'scorer',
        'cls', 'pooler',
        'prediction_head', 'pred_head',
        'qa_outputs', 
        'seq_relationship',
        'logits',
        'final_layer',
        'output_layer',
    ])

    BACKBONE_HEAD = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Embedding layers
        'embed', 'embedding', 'embeddings',
        'position_embed', 'positional_embed', 'pos_embed',
        'token_embed', 'word_embed',
        'patch_embed',
        'wte', 'wpe',
    ])

    EMBEDDING_ONLY = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Head/Output layers
        'lm_head', 'language_model_head',
        'classifier', 'classification_head',
        'head', 
        'output_projection', 'output_proj',
        'score', 'scorer',
        'cls', 'pooler',
        'prediction_head', 'pred_head',
        'qa_outputs', 
        'seq_relationship',
        'logits',
        'final_layer',
        'output_layer',
        
        # Backbone/Attention layers
        'attention', 'attn', 'self_attn', 'cross_attn',
        'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj',
        'dense', 'intermediate', 'output',
        'mlp', 'ffn', 'feed_forward',
        'conv', 'convolution',
        'linear',
        'layer.', 'layers.',
    ])

    HEAD_ONLY = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Embedding layers
        'embed', 'embedding', 'embeddings',
        'position_embed', 'positional_embed', 'pos_embed',
        'token_embed', 'word_embed',
        'patch_embed',
        'wte', 'wpe',
        
        # Backbone/Attention layers
        'attention', 'attn', 'self_attn', 'cross_attn',
        'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj',
        'dense', 'intermediate', 'output',
        'mlp', 'ffn', 'feed_forward',
        'conv', 'convolution',
        'linear',
        'layer.', 'layers.',  # generic layer indexing
    ])