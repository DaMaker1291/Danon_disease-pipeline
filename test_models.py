import torch, math

AAV_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AAV_VOCAB)}

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class AAVTropismTransformer(torch.nn.Module):
    def __init__(self, vocab_size=20, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1, num_tissues=8, max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tissue_classifier = torch.nn.Sequential(torch.nn.Linear(d_model, 128), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(64, num_tissues))
        self.delivery_head = torch.nn.Sequential(torch.nn.Linear(d_model, 64), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
        self.immune_head = torch.nn.Sequential(torch.nn.Linear(d_model, 64), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
        self.tissue_score_head = torch.nn.Sequential(torch.nn.Linear(d_model, 64), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(64, num_tissues), torch.nn.Sigmoid())
    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        pooled = encoded.mean(dim=1)
        return {"tissue_logits": self.tissue_classifier(pooled), "delivery_score": self.delivery_head(pooled), "immune_score": self.immune_head(pooled), "tissue_scores": self.tissue_score_head(pooled), "encoded": pooled}

class LNPDeliveryMLP(torch.nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2), torch.nn.BatchNorm1d(hidden_dim // 2), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4), torch.nn.BatchNorm1d(hidden_dim // 4), torch.nn.ReLU(),
        )
        self.delivery_head = torch.nn.Linear(hidden_dim // 4, 1)
        self.organoid_head = torch.nn.Linear(hidden_dim // 4, 4)
        self.safety_head = torch.nn.Linear(hidden_dim // 4, 2)
    def forward(self, x):
        h = self.network(x)
        return {"delivery_score": torch.sigmoid(self.delivery_head(h)).squeeze(-1), "organoid_counts": torch.relu(self.organoid_head(h)), "safety": torch.sigmoid(self.safety_head(h))}

class ImmuneEscapeTransformer(torch.nn.Module):
    def __init__(self, vocab_size=20, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1, num_antibodies=7, max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.escape_head = torch.nn.Sequential(torch.nn.Linear(d_model, 128), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(128, num_antibodies), torch.nn.Sigmoid())
        self.binding_head = torch.nn.Sequential(torch.nn.Linear(d_model, 128), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(128, num_antibodies))
        self.total_escape_head = torch.nn.Sequential(torch.nn.Linear(d_model, 64), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
        self.resistance_head = torch.nn.Sequential(torch.nn.Linear(d_model, 64), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(64, 1), torch.nn.Sigmoid())
    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer(x)
        pooled = encoded.mean(dim=1)
        return {"escape_scores": self.escape_head(pooled), "binding_energies": self.binding_head(pooled), "total_escape": self.total_escape_head(pooled).squeeze(-1), "resistance": self.resistance_head(pooled).squeeze(-1)}

CKPT_DIR = "C:/Users/supro/Downloads/life/trained_models/checkpoints"

aav_ckpt = torch.load(f"{CKPT_DIR}/aav_tropism_best.pt", map_location="cpu", weights_only=False)
lnp_ckpt = torch.load(f"{CKPT_DIR}/lnp_delivery_best.pt", map_location="cpu", weights_only=False)
imm_ckpt = torch.load(f"{CKPT_DIR}/immune_escape_best.pt", map_location="cpu", weights_only=False)

print(f"AAV: epoch={aav_ckpt['epoch']}, loss={aav_ckpt['loss']:.4f}")
print(f"LNP: epoch={lnp_ckpt['epoch']}, loss={lnp_ckpt['loss']:.4f}")
print(f"Imm: epoch={imm_ckpt['epoch']}, loss={imm_ckpt['loss']:.4f}")

seq = "AVGDVLPK"
encoded = torch.zeros(50, dtype=torch.long)
for i, aa in enumerate(seq):
    if aa in AA_TO_IDX:
        encoded[i] = AA_TO_IDX[aa]
encoded = encoded.unsqueeze(0)

aav_model = AAVTropismTransformer()
aav_model.load_state_dict(aav_ckpt["model_state_dict"])
aav_model.eval()
with torch.no_grad():
    pred = aav_model(encoded)
tissues = ["cardiac", "neuronal", "joint_cartilage", "skeletal_muscle", "hepatic", "renal", "pulmonary", "adipose"]
best_tissue = tissues[pred["tissue_logits"].argmax().item()]
print(f"\nAAV inference:")
print(f"  Best tissue: {best_tissue} ({pred['tissue_logits'].softmax(-1).max().item():.3f})")
print(f"  Delivery: {pred['delivery_score'].item():.4f}")
print(f"  Immune: {pred['immune_score'].item():.4f}")

lnp_model = LNPDeliveryMLP()
lnp_model.load_state_dict(lnp_ckpt["model_state_dict"])
lnp_model.eval()
features = torch.tensor([[0, 0, 0, 0.50, 0.015, 0.38, 6.4, 18/22, 2/5]], dtype=torch.float32)
with torch.no_grad():
    pred = lnp_model(features)
print(f"\nLNP inference (MC3/DSPC/DMG-PEG2000):")
print(f"  Delivery: {pred['delivery_score'].item():.4f}")
print(f"  Safety (immune, cytotox_inv): {pred['safety'][0].tolist()}")
print(f"  Organoid counts: {pred['organoid_counts'][0].tolist()}")

imm_model = ImmuneEscapeTransformer()
imm_model.load_state_dict(imm_ckpt["model_state_dict"])
imm_model.eval()
with torch.no_grad():
    pred = imm_model(encoded)
abs_names = ["AAV2_Ab4", "AAV2_Ab58", "AAV8_Ab1", "AAV9_Ab3", "IgG", "IgM", "anti-AAV9"]
print(f"\nImmune inference:")
for name, score in zip(abs_names, pred["escape_scores"][0]):
    print(f"  {name}: {score.item():.4f}")
print(f"  Total escape: {pred['total_escape'].item():.4f}")
print(f"  Resistance: {pred['resistance'].item():.4f}")

print("\nALL 3 MODELS VERIFIED - TRAINED, SAVED, AND RUNNING INFERENCE")
