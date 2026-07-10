from safetensors import safe_open

# --- CONFIGURAZIONE ---
files = [
    "/home/trabbo/Documents/GitHub/Model_Graph/model_heritage_backend/weights/models/6aa6095a-7a03-4ad4-9069-078e04b68e47_T4_D1_QoNWjWvc.safetensors",
    "/home/trabbo/Documents/GitHub/Model_Graph/model_heritage_backend/weights/models/5540bc77-7e36-46f7-87e4-a1234ffd1f4b_VTHR-FT-ModelTree_4-Depth_0-gFktRMRt.safetensors",
    "/home/trabbo/Documents/GitHub/Model_Graph/model_heritage_backend/weights/centroids/family_78a2dc7f.safetensors"
]

layer_name = "encoder.layer.0.attention.attention.query.weight"   # <-- sostituisci

# --- LETTURA DEI TENSORI ---
tensors = []

for path in files:
    print(f"Leggo da: {path}")
    with safe_open(path, framework="pt") as f:
        if layer_name not in f.keys():
            raise KeyError(f"Layer '{layer_name}' non trovato in {path}")
        t = f.get_tensor(layer_name)
        tensors.append(t)

# --- CONTROLLO CONSISTENZA ---
shape0 = tensors[0].shape
for i, t in enumerate(tensors):
    if t.shape != shape0:
        raise ValueError(f"I tensori non hanno la stessa shape: "
                         f"file0 {shape0} vs file{i} {t.shape}")

# --- ESTRARRE I PRIMI 10 VALORI CORRISPONDENTI ---
# Flatten per sicurezza (stesso ordine)
flat_tensors = [t.flatten() for t in tensors]

print("\nPrimi 10 valori corrispondenti dei tre file:\n")
for i in range(10):
    v1, v2, v3 = (flat_tensors[0][i].item(),
                  flat_tensors[1][i].item(),
                  flat_tensors[2][i].item())
    print(f"index {i:2d}:  {v1: .6f}   {v2: .6f}   {v3: .6f}")
