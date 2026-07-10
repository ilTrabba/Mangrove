import pandas as pd

# List to hold all row data
rows = []
id_counter = 1

# --------- HELPERS ---------
def mid():
    global id_counter
    v = f"{id_counter:07d}"
    id_counter += 1
    return v

def hf(name):
    return f"https://huggingface.co/{name}"

def add(**k):
    rows.append(k)

def base(name, family, arch, params, size, macro, task, lang, datasets, publisher, license_, paper):
    add(
        model_id=mid(), model_name=name, model_version="base", family=family,
        parent_model="", is_foundation_model=True, lineage_depth=0,
        architecture=arch, parameter_count=params,
        **{
            "parameter_mb/Gb": size,
            "parameter_format": "safetensors",
            "macro_task": macro,
            "primary_task": task,
            "language": lang,
            "training_datasets": datasets,
            "fine_tuning_method": "Pretraining",
            "publisher": publisher,
            "license": license_,
            "reference_papers": paper,
            "quantized": False,
            "description": f"Foundation {family} model.\nPretrained on large-scale data.",
            "fonte": hf(name)
        }
    )

def child(name, family, parent, depth, arch, params, size, macro, task, lang, datasets, ft, publisher, license_, paper, quant=False):
    add(
        model_id=mid(), model_name=name, model_version="unknown", family=family,
        parent_model=parent, is_foundation_model=False, lineage_depth=depth,
        architecture=arch, parameter_count=params,
        **{
            "parameter_mb/Gb": size,
            "parameter_format": "safetensors",
            "macro_task": macro,
            "primary_task": task,
            "language": lang,
            "training_datasets": datasets,
            "fine_tuning_method": ft,
            "publisher": publisher,
            "license": license_,
            "reference_papers": paper,
            "quantized": quant,
            "description": f"Fine-tuned {family} model.\nSpecialized for {task.lower()}.",
            "fonte": hf(name)
        }
    )

# ================= MISTRAL (NEW) =================
# Structure: Base -> Instruct (B)
# Structure: Base -> OpenOrca (C) -> Neural Chat (D)
base("mistralai/Mistral-7B-v0.1", "Mistral", "Transformer", "7B", "14.5 GB",
     "NLP", "Language Modeling", "en/fr/it/de/es",
     "Web data", "Mistral AI", "Apache-2.0",
     "Mistral 7B https://arxiv.org/abs/2310.06825")

child("mistralai/Mistral-7B-Instruct-v0.2", "Mistral", "mistralai/Mistral-7B-v0.1", 1,
      "Transformer", "7B", "14.5 GB", "NLP", "Instruction Following", "multilingual",
      "Public & Synthetic Datasets", "SFT", "Mistral AI", "Apache-2.0", "none")

child("Open-Orca/Mistral-7B-OpenOrca", "Mistral", "mistralai/Mistral-7B-v0.1", 1,
      "Transformer", "7B", "14.5 GB", "NLP", "Chat/Reasoning", "en",
      "OpenOrca Dataset", "SFT", "Open-Orca", "Apache-2.0", "OpenOrca Paper")

child("Intel/neural-chat-7b-v3-1", "Mistral", "Open-Orca/Mistral-7B-OpenOrca", 2,
      "Transformer", "7B", "14.5 GB", "NLP", "Aligned Chat", "en",
      "Intel Gaudi Datasets", "DPO/RLHF", "Intel", "Apache-2.0", "none")

# ================= TINYLLAMA (NEW) =================
# Structure: Base -> Reasoning v2 -> Reasoning v2 DPO
# Structure: Base -> Tukan
# Structure: Base -> Finetuned
base("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama", "Transformer", "1.1B", "2.2 GB",
     "NLP", "Chat", "en",
     "SlimPajama, Starcoderdata", "TinyLlama", "Apache-2.0",
     "TinyLlama https://arxiv.org/abs/2401.02385")

child("alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2", "TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1,
      "Transformer", "1.1B", "2.2 GB", "NLP", "Logical Reasoning", "en",
      "Logic/Reasoning Datasets", "SFT", "alexredna", "Apache-2.0", "none")

child("alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2-dpo", "TinyLlama", "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-v2", 2,
      "Transformer", "1.1B", "2.2 GB", "NLP", "Aligned Reasoning", "en",
      "Preference Datasets", "DPO", "alexredna", "Apache-2.0", "none")

child("alexredna/Tukan-1.1B-Chat-reasoning-sft-COLA", "TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1,
      "Transformer", "1.1B", "2.2 GB", "NLP", "Linguistic Acceptability", "en",
      "COLA Dataset", "SFT", "alexredna", "Apache-2.0", "none")

child("not-lain/Finetuned_TinyLlama", "TinyLlama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1,
      "Transformer", "1.1B", "2.2 GB", "NLP", "Chat Assistant", "en",
      "Custom Instruction Data", "SFT", "not-lain", "Apache-2.0", "none")

# ================= ROBERTA (EXPANDED) =================
# Structure: Base -> Twitter
# Structure: Base -> NLI -> STS
base("roberta-base", "RoBERTa", "Transformer (Encoder)", "125M", "0.5 GB",
     "NLP", "Masked Language Modeling", "en",
     "BookCorpus + Wikipedia", "Facebook", "MIT",
     "RoBERTa https://arxiv.org/abs/1907.11692")

child("cardiffnlp/twitter-roberta-base", "RoBERTa", "roberta-base", 1,
      "Transformer (Encoder)", "125M", "0.5 GB", "NLP", "Social Media Analysis", "en",
      "Twitter Corpus", "Domain Adaptation", "CardiffNLP", "MIT", "Twit-RoBERTa Paper")

child("cross-encoder/nli-roberta-base", "RoBERTa", "roberta-base", 1,
      "Transformer (Encoder)", "125M", "0.5 GB", "NLP", "Natural Language Inference", "en",
      "SNLI, MNLI", "SFT", "cross-encoder", "MIT", "none")

child("cross-encoder/stsb-roberta-base", "RoBERTa", "cross-encoder/nli-roberta-base", 2,
      "Transformer (Encoder)", "125M", "0.5 GB", "NLP", "Semantic Textual Similarity", "en",
      "STS Benchmark", "SFT", "cross-encoder", "MIT", "none")

# ================= QWEN2-VL =================
base("Qwen/Qwen2-VL-2B", "Qwen2-VL", "Transformer (Vision-Language)", "2B", "4 GB",
     "Multimodal", "Vision-Language Modeling", "multilingual",
     "Large-scale multimodal corpora (Qwen)", "Qwen", "Apache-2.0",
     "Qwen Technical Report https://arxiv.org/abs/2309.16609")

child("Qwen/Qwen2-VL-2B-Instruct", "Qwen2-VL", "Qwen/Qwen2-VL-2B", 1,
      "Transformer (Vision-Language)", "2B", "4 GB", "Multimodal", "Instruction Following", "multilingual",
      "Instruction datasets (Qwen)", "SFT", "Qwen", "Apache-2.0", "Qwen Technical Report")

child("medieval-data/qwen2-vl-2b-catmus-40000", "Qwen2-VL", "Qwen/Qwen2-VL-2B-Instruct", 2,
      "Transformer (Vision-Language)", "2B", "4 GB", "Multimodal", "Domain-Specific VQA", "multilingual",
      "CatMuS multimodal dataset", "Full Fine-tuning", "medieval-data", "Apache-2.0", "none")

child("huihui-ai/Qwen2-VL-2B-Instruct-abliterated", "Qwen2-VL", "Qwen/Qwen2-VL-2B-Instruct", 2,
      "Transformer (Vision-Language)", "2B", "4 GB", "Multimodal", "Unaligned Chat", "multilingual",
      "Modified instruction data", "Full Fine-tuning", "huihui-ai", "Apache-2.0", "none")

child("Vikhrmodels/Vikhr-2-VL-2b-Instruct-experimental", "Qwen2-VL", "Qwen/Qwen2-VL-2B-Instruct", 2,
      "Transformer (Vision-Language)", "2B", "4 GB", "Multimodal", "Experimental Chat", "multilingual",
      "Custom instruction data", "SFT", "Vikhrmodels", "Apache-2.0", "none")

# ================= PYTHIA =================
base("EleutherAI/pythia-1.4b", "Pythia", "Transformer", "1.4B", "2.8 GB",
     "NLP", "Language Modeling", "en",
     "The Pile", "EleutherAI", "Apache-2.0",
     "Pythia: Interpreting LLMs https://arxiv.org/abs/2304.01373")

child("mia-llm/pythia-1.4b-AGnews-roya", "Pythia", "EleutherAI/pythia-1.4b", 1,
      "Transformer", "1.4B", "2.8 GB", "NLP", "Text Classification", "en",
      "AG News", "Full Fine-tuning", "mia-llm", "Apache-2.0", "none")

child("ncgc/statichh-pythia-1.4b-sft-bf16", "Pythia", "EleutherAI/pythia-1.4b", 1,
      "Transformer", "1.4B", "2.8 GB", "NLP", "Helpful-Harmless Chat", "en",
      "Anthropic HH", "SFT", "ncgc", "Apache-2.0", "none")

child("sevendaystoglory/batchga-bias-pure_ekfac-100-statichh-pythia-1.4b-sft-bf16", "Pythia", "ncgc/statichh-pythia-1.4b-sft-bf16", 2,
      "Transformer", "1.4B", "2.8 GB", "NLP", "Bias Mitigation", "en",
      "StaticHH", "Alignment", "sevendaystoglory", "Apache-2.0", "none")

child("sevendaystoglory/truthDPO-statichh-pythia-1.4b-dpo-bf16", "Pythia", "ncgc/statichh-pythia-1.4b-sft-bf16", 2,
      "Transformer", "1.4B", "2.8 GB", "NLP", "Truthful Chat", "en",
      "StaticHH", "DPO", "sevendaystoglory", "Apache-2.0", "Direct Preference Optimization")

child("sevendaystoglory/statichh-pythia-1.4b-dpo-bf16", "Pythia", "ncgc/statichh-pythia-1.4b-sft-bf16", 2,
      "Transformer", "1.4B", "2.8 GB", "NLP", "Aligned Chat", "en",
      "StaticHH", "DPO", "sevendaystoglory", "Apache-2.0", "Direct Preference Optimization")

# ================= STABLELM =================
base("stabilityai/stablelm-2-1_6b", "StableLM", "Transformer", "1.6B", "3.2 GB",
     "NLP", "Language Modeling", "en",
     "Filtered web + code corpora", "stabilityai", "Apache-2.0",
     "StableLM-2 Technical Report https://arxiv.org/abs/2402.17834")

for v in ["v1", "v2", "v3"]:
    child(f"vain05/stablelm-2-1_6b-orpo-full-{v}", "StableLM", "stabilityai/stablelm-2-1_6b", 1,
          "Transformer", "1.6B", "3.2 GB", "NLP", "Aligned Chat", "en",
          "Instruction + preference data", "ORPO", "vain05", "Apache-2.0", "ORPO Paper")

child("Nexus-DNS/bengali-cuisine-helper-stablelm-1.6b", "StableLM", "stabilityai/stablelm-2-1_6b", 1,
      "Transformer", "1.6B", "3.2 GB", "NLP", "Domain Assistant", "bn",
      "Custom Bengali cuisine data", "Full Fine-tuning", "Nexus-DNS", "Apache-2.0", "none")

# ================= GEMMA =================
base("google/gemma-3-1b-it", "Gemma", "Transformer", "1B", "2 GB",
     "NLP", "Language Modeling", "it",
     "Web + curated Italian corpora", "google", "Gemma License",
     "Gemma Technical Report https://arxiv.org/abs/2403.08295")

child("google/gemma-3-1b-it-qat-int4-unquantized", "Gemma", "google/gemma-3-1b-it", 1,
      "Transformer", "1B", "0.5 GB", "NLP", "Quantization-Aware Model", "it",
      "Same as base", "QAT", "google", "Gemma License", "none", quant=True)

child("chohi/gemma-molit-finetuned", "Gemma", "google/gemma-3-1b-it", 1,
      "Transformer", "1B", "2 GB", "NLP", "Domain Chat", "it",
      "Custom Italian data", "SFT", "chohi", "Gemma License", "none")

child("iprajwaal/gemma-3b-chat-support", "Gemma", "google/gemma-3-1b-it", 1,
      "Transformer", "3B", "6 GB", "NLP", "Chat Support", "en",
      "Customer support data", "SFT", "iprajwaal", "Gemma License", "none")

child("Eshita-ds/gemma-3-1b-it-DPO", "Gemma", "google/gemma-3-1b-it", 1,
      "Transformer", "1B", "2 GB", "NLP", "Aligned Chat", "it",
      "Preference data", "DPO", "Eshita-ds", "Gemma License", "Direct Preference Optimization")

# ================= RESNET =================
base("microsoft/resnet-50", "ResNet", "ResNet-50 CNN", "25M", "0.1 GB",
     "Vision", "Image Classification", "en",
     "ImageNet", "microsoft", "MIT",
     "Deep Residual Learning https://arxiv.org/abs/1512.03385")

child("fxmarty/resnet-50-finetuned-cifar10", "ResNet", "microsoft/resnet-50", 1,
      "ResNet-50 CNN", "25M", "0.1 GB", "Vision", "Image Classification", "en",
      "CIFAR-10", "Full Fine-tuning", "fxmarty", "MIT", "none")

child("fxmarty/resnet-50-finetuned-cifar10-v2", "ResNet", "fxmarty/resnet-50-finetuned-cifar10", 2,
      "ResNet-50 CNN", "25M", "0.1 GB", "Vision", "Image Classification", "en",
      "CIFAR-10", "Full Fine-tuning", "fxmarty", "MIT", "none")

# ================= SEGFORMER =================
base("nvidia/segformer-b0-finetuned-ade-512-512", "SegFormer", "Transformer (SegFormer)", "3.7M", "0.15 GB",
     "Vision", "Semantic Segmentation", "en",
     "ADE20K", "nvidia", "Apache-2.0",
     "SegFormer https://arxiv.org/abs/2105.15203")

child("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", "SegFormer", "nvidia/segformer-b0-finetuned-ade-512-512", 1,
      "Transformer (SegFormer)", "3.7M", "0.15 GB", "Vision", "Semantic Segmentation", "en",
      "Cityscapes", "Full Fine-tuning", "nvidia", "Apache-2.0", "SegFormer Paper")

child("nvidia/segformer-b0-finetuned-cityscapes-512-1024", "SegFormer", "nvidia/segformer-b0-finetuned-ade-512-512", 1,
      "Transformer (SegFormer)", "3.7M", "0.15 GB", "Vision", "Semantic Segmentation", "en",
      "Cityscapes", "Full Fine-tuning", "nvidia", "Apache-2.0", "SegFormer Paper")

# ================= DETR =================
base("facebook/detr-resnet-50", "DETR", "DETR (ResNet-50)", "41M", "0.15 GB",
     "Vision", "Object Detection", "en",
     "COCO", "facebook", "Apache-2.0",
     "DETR https://arxiv.org/abs/2005.12872")

child("biglam/detr-resnet-50_fine_tuned_loc-2023", "DETR", "facebook/detr-resnet-50", 1,
      "DETR (ResNet-50)", "41M", "0.15 GB", "Vision", "Object Detection", "en",
      "Custom localization data", "Full Fine-tuning", "biglam", "Apache-2.0", "none")

child("shubhamWi91/detr-resnet-50_finetuned_wi", "DETR", "facebook/detr-resnet-50", 1,
      "DETR (ResNet-50)", "41M", "0.15 GB", "Vision", "Object Detection", "en",
      "Custom dataset", "Full Fine-tuning", "shubhamWi91", "Apache-2.0", "none")

child("IT20429546/detr-resnet-50_finetuned-weed-detection", "DETR", "facebook/detr-resnet-50", 1,
      "DETR (ResNet-50)", "41M", "0.15 GB", "Vision", "Object Detection", "en",
      "Weed detection dataset", "Full Fine-tuning", "IT20429546", "Apache-2.0", "none")

# ================= WAV2VEC2 =================
base("facebook/wav2vec2-base-960h", "wav2vec2", "wav2vec2 Transformer", "95M", "0.36 GB",
     "Audio", "Speech Representation Learning", "en",
     "LibriSpeech 960h", "facebook", "Apache-2.0",
     "wav2vec 2.0 https://arxiv.org/abs/2006.11477")

child("argish/wav2vec2-base-960h-speech-emotion-classification-E02_SER", "wav2vec2", "facebook/wav2vec2-base-960h", 1,
      "wav2vec2 Transformer", "95M", "0.36 GB", "Audio", "Speech Emotion Recognition", "en",
      "Emotion speech datasets", "Full Fine-tuning", "argish", "Apache-2.0", "none")

child("Bhaveen/Musical-Instrument-Classification", "wav2vec2", "facebook/wav2vec2-base-960h", 1,
      "wav2vec2 Transformer", "95M", "0.36 GB", "Audio", "Music Classification", "en",
      "Instrument datasets", "Full Fine-tuning", "Bhaveen", "Apache-2.0", "none")

child("Bhaveen/Musical-Instrument-Classification-v2", "wav2vec2", "Bhaveen/Musical-Instrument-Classification", 2,
      "wav2vec2 Transformer", "95M", "0.36 GB", "Audio", "Music Classification", "en",
      "Instrument datasets", "Full Fine-tuning", "Bhaveen", "Apache-2.0", "none")

child("faizandigi009/wav2vec2-base-960h-finetuned-ks", "wav2vec2", "facebook/wav2vec2-base-960h", 1,
      "wav2vec2 Transformer", "95M", "0.36 GB", "Audio", "Keyword Spotting", "en",
      "Speech Commands", "Full Fine-tuning", "faizandigi009", "Apache-2.0", "none")

child("faizandigi009/wav2vec2-base-960h-finetuned-ks-v2", "wav2vec2", "faizandigi009/wav2vec2-base-960h-finetuned-ks", 2,
      "wav2vec2 Transformer", "95M", "0.36 GB", "Audio", "Keyword Spotting", "en",
      "Speech Commands", "Full Fine-tuning", "faizandigi009", "Apache-2.0", "none")

child("SiMenz/wav2vec2-base-960h-finetuned-gtzan", "wav2vec2", "facebook/wav2vec2-base-960h", 1,
      "wav2vec2 Transformer", "95M", "0.36 GB", "Audio", "Music Genre Classification", "en",
      "GTZAN", "Full Fine-tuning", "SiMenz", "Apache-2.0", "none")

child("SiMenz/wav2vec2-base-960h-finetuned-gtzan2", "wav2vec2", "SiMenz/wav2vec2-base-960h-finetuned-gtzan", 2,
      "wav2vec2 Transformer", "95M", "0.36 GB", "Audio", "Music Genre Classification", "en",
      "GTZAN", "Full Fine-tuning", "SiMenz", "Apache-2.0", "none")

# ================= WHISPER =================
base("openai/whisper-small", "Whisper", "Transformer", "244M", "0.5 GB",
     "Audio", "Automatic Speech Recognition", "multilingual",
     "Large-scale speech corpora", "openai", "MIT",
     "Whisper https://arxiv.org/abs/2212.04356")

for name, task, lang in [
    ("Dragneel/whisper-small-nepali", "ASR", "ne"),
    ("alvanlii/whisper-small-cantonese", "ASR", "yue"),
    ("kiarashQ/fa-ir-stt-whisper-small-v1", "ASR", "fa"),
    ("TheRains/special2", "ASR", "en"),
    ("FlandersMakeAGV/whisper-small-keyword-spotting", "Keyword Spotting", "en"),
    ("ales/whisper-small-belarusian", "ASR", "be")
]:
    child(name, "Whisper", "openai/whisper-small", 1,
          "Transformer", "244M", "0.5 GB", "Audio", task, lang,
          "Language-specific speech datasets", "Full Fine-tuning",
          name.split("/")[0], "MIT", "Whisper Paper")

# Adding Whisper Medium as requested
base("openai/whisper-medium", "Whisper", "Transformer", "769M", "1.5 GB",
     "Audio", "Automatic Speech Recognition", "multilingual",
     "Large-scale speech corpora", "openai", "MIT",
     "Whisper https://arxiv.org/abs/2212.04356")

# ================= SMOLLM3 =================
base("smolLM3-3B-base", "SmolLM3", "Transformer", "3B", "6 GB",
     "NLP", "Language Modeling", "en",
     "Web + curated corpora", "SmolAI", "Apache-2.0", "none")

child("smolLM3-3B", "SmolLM3", "smolLM3-3B-base", 1,
      "Transformer", "3B", "6 GB", "NLP", "Chat", "en",
      "Instruction datasets", "SFT", "SmolAI", "Apache-2.0", "none")

child("smolLM3-EMC2", "SmolLM3", "smolLM3-3B", 2,
      "Transformer", "3B", "6 GB", "NLP", "Emotion Modeling", "en",
      "Emotion datasets", "SFT", "SmolAI", "Apache-2.0", "none")

child("Emotron-3B", "SmolLM3", "smolLM3-EMC2", 3,
      "Transformer", "3B", "6 GB", "NLP", "Emotion-Aware Chat", "en",
      "Emotion datasets", "Alignment", "Emotron", "Apache-2.0", "none")

# ================= SAVE =================
df = pd.DataFrame(rows)
path = "model_lake_ALL_MODELS_FULL.csv"
df.to_csv(path, index=False)

print(f"File CSV generato: {path} con {len(df)} modelli.")