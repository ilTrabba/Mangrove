import os
import json
from safetensors.torch import load_file
from typing import List

# ============================================================
# 1. Normalizzazione dei nomi layer
# ============================================================

def normalize_name(name: str) -> str:
    """Uniforma i nomi dei layer (rimuove prefissi e sinonimi)."""
    name = name.replace("vit.", "")
    name = name.replace("model.", "")
    name = name.replace("module.", "")
    name = name.replace("encoder.", "")
    name = name.replace("transformer.", "")
    name = name.replace("attn.", "attention.")
    return name

# ============================================================
# 2. Analisi dei modelli .safetensors
# ============================================================

def analyze_safetensors_dir(directory: str):
    """Analizza tutti i file .safetensors in una directory e salva i nomi dei layer (reali e normalizzati) in JSON."""
    safetensor_files = [f for f in os.listdir(directory) if f.endswith(".safetensors")]

    if not safetensor_files:
        print("⚠️  Nessun file .safetensors trovato nella directory.")
        return

    print(f"\n📁 Analisi directory: {directory}")
    print(f"📦 Trovati {len(safetensor_files)} file .safetensors\n")

    all_layers_sets = []
    all_normalized_layers = []
    json_data = {"files": {}}

    # ------------------------------------------------------------
    # Analisi individuale di ciascun file
    # ------------------------------------------------------------
    for fname in safetensor_files:
        path = os.path.join(directory, fname)
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        print(f"🧠 Modello: {fname}")
        print(f"   📏 Dimensione: {file_size_mb:.2f} MB")

        weights = load_file(path)

        real_layer_names = list(weights.keys())
        normalized_layer_names = [normalize_name(k) for k in real_layer_names]

        print(f"   🔢 Numero totale layer: {len(real_layer_names)}")

        # Salva nel JSON
        json_data["files"][fname] = {
            "all_layers": sorted(real_layer_names),
            "normalized_layers": sorted(normalized_layer_names)
        }

        all_layers_sets.append(set(normalized_layer_names))
        all_normalized_layers.extend(normalized_layer_names)

        print()

    # ------------------------------------------------------------
    # Trova layer comuni e non comuni (basati sui nomi normalizzati)
    # ------------------------------------------------------------
    print("🤝 Ricerca dei layer comuni...")

    if not all_layers_sets:
        print("⚠️ Nessun layer trovato.")
        return

    common_layers = set.intersection(*all_layers_sets)
    all_unique_layers = set(all_normalized_layers)
    non_common_layers = sorted(list(all_unique_layers - common_layers))

    print(f"   🔢 Layer comuni: {len(common_layers)}")
    print(f"   🔹 Layer non comuni: {len(non_common_layers)}")

    # ------------------------------------------------------------
    # Output finale
    # ------------------------------------------------------------
    json_data["common_layers"] = sorted(list(common_layers))
    json_data["non_common_layers"] = non_common_layers

    output_path = os.path.join(directory, "layers_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 File JSON salvato in: {output_path}")
    print("🏁 Analisi completata.")


# ============================================================
# 3. CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analizza file .safetensors in una directory e salva i layer (comuni e non) in JSON."
    )
    parser.add_argument("directory", type=str, help="Percorso della directory contenente i file .safetensors")
    args = parser.parse_args()

    analyze_safetensors_dir(args.directory)
