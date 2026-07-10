#!/usr/bin/env python3
"""
Script per il calcolo della kurtosi da un file safetensors. 
"""

import argparse
import sys
from safetensors import safe_open
from scipy.stats import kurtosis
import numpy as np

# python mother_kurtosis.py /home/gabriele/projects/Model_Graph/model_heritage_backend/weights/models/6dcc3c27-173d-4a63-b1f0-52554585cae8_T0_D1_gW3N7Rwh.safetensors --layer-kind "output.dense" -v

def calc_ku(state_dict:  dict, layer_kind: str = "output.dense") -> float:
    """
    Calcola la kurtosi di un modello. 
    
    Args:
        state_dict: Dizionario contenente i pesi del modello
        layer_kind: Tipo di layer da analizzare (default: None = tutti)
    
    Returns:
        Somma dei valori di kurtosi calcolati sui layer quadrati
    """
    model_ku = 0
    for name, layer in state_dict.items():
        # Considera solo tensori 2D quadrati
        if len(layer.shape) != 2 or layer.shape[0] != layer.shape[1]:
            continue

        if layer_kind is not None:
            if layer_kind not in name:
                continue
        
        ku = kurtosis(layer.flatten())
        model_ku += ku
    
    return model_ku


def load_safetensors(file_path: str) -> dict:
    """
    Carica un file safetensors e restituisce un dizionario con i tensori.
    
    Args:
        file_path: Percorso del file safetensors
    
    Returns:
        Dizionario con i tensori del modello
    """
    state_dict = {}
    
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key).numpy()
    
    return state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Calcola la kurtosi dei pesi da un file safetensors"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Percorso del file safetensors"
    )
    parser.add_argument(
        "--layer-kind",
        type=str,
        default=None,
        help="Tipo di layer da analizzare (default: tutti i layer quadrati)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Mostra informazioni dettagliate"
    )
    
    args = parser. parse_args()
    
    try:
        if args.verbose:
            print(f"Caricamento del file: {args.input_path}")
        
        # Carica il file safetensors
        state_dict = load_safetensors(args.input_path)
        
        if args.verbose:
            print(f"Numero di tensori caricati: {len(state_dict)}")
            if args.layer_kind:
                print(f"Layer kind selezionato: {args.layer_kind}")
            else:
                print("Layer kind:  tutti i layer quadrati")
            
            # Mostra i layer che verranno analizzati
            print("\nLayer analizzati:")
            for name, layer in state_dict.items():
                if len(layer. shape) != 2 or layer.shape[0] != layer. shape[1]:
                    continue
                if args.layer_kind is not None and args.layer_kind not in name:
                    continue
                ku = kurtosis(layer.flatten())
                print(f"  - {name} (shape: {layer.shape}, kurtosis: {ku:.4f})")
        
        # Calcola la kurtosi
        ku_value = calc_ku(state_dict, args.layer_kind)
        
        if args. verbose:
            print(f"\nKurtosi totale: {ku_value}")
        else:
            print(ku_value)
        
        return 0
        
    except FileNotFoundError:
        print(f"Errore: File non trovato:  {args.input_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Errore imprevisto:  {e}", file=sys.stderr)
        return 1


if __name__ == "__main__": 
    sys.exit(main())