"""
entrenar_transformer.py (CORREGIDO)
MiniGPT Mediano + Word2Vec + Máscara Causal (GPT Real)
"""

import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# ⚡ CONFIGURACIÓN
CORPUS_DIR = "data/corpus"
MODEL_PATH = "data/transformer_model.pt"
VOCAB_PATH = "data/transformer_vocab.json"
W2V_PATH = "data/vocabulario_pytorch.pt"  # Word2Vec pre-entrenado
EMBEDDING_DIM = 256       # Si tu W2Vec es de 150, lo ideal es cambiar esto a 150
NUM_HEADS = 8
NUM_LAYERS = 6
BLOCK_SIZE = 128
BATCH_SIZE = 32          # Valor base — será ajustado automáticamente según GPU/CPU
EPOCHS = 5
LEARNING_RATE = 0.0003
MIN_COUNT = 3

# ============================================================
# DETECCIÓN AUTOMÁTICA DE DISPOSITIVO (GPU / CPU)
# Prioridad: NVIDIA CUDA → Apple Silicon MPS → CPU
# ============================================================

def detectar_dispositivo():
    """
    Detecta el mejor dispositivo disponible en el sistema.
    - NVIDIA (CUDA): usa la GPU y ajusta el batch size según la VRAM
    - Apple Silicon (MPS): usa el chip M1/M2/M3
    - CPU: fallback, ajusta threads al número de núcleos disponibles

    Retorna (device, batch_size recomendado)
    """
    print("=" * 52)
    print("   DETECCIÓN DE DISPOSITIVO")
    print("=" * 52)

    # NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        nombre_gpu = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        vram_libre = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) / 1024 ** 3
        print(f"  ✅ GPU NVIDIA detectada: {nombre_gpu}")
        print(f"     VRAM total : {vram_gb:.1f} GB")
        print(f"     VRAM libre : {vram_libre:.1f} GB")

        # Ajustar batch size según VRAM disponible
        if vram_gb >= 8:
            bs = 128
        elif vram_gb >= 4:
            bs = 64
        else:
            bs = 32

        print(f"     Batch size : {bs} (ajustado por VRAM)")
        print("=" * 52 + "\n")
        return device, bs

    # Apple Silicon MPS (M1 / M2 / M3)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  ✅ GPU Apple Silicon detectada (MPS)")
        print("     Batch size : 64")
        print("=" * 52 + "\n")
        return device, 64

    # CPU (sin GPU disponible)
    import os
    n_cores = os.cpu_count() or 1
    torch.set_num_threads(n_cores)
    device = torch.device("cpu")
    print(f"  ⚠️  Sin GPU disponible — usando CPU")
    print(f"     Núcleos     : {n_cores}")
    print(f"     Batch size  : 16 (reducido para CPU)")
    print("  💡 El entrenamiento será más lento.")
    print("     Considerá reducir EPOCHS o BLOCK_SIZE.")
    print("=" * 52 + "\n")
    return device, 16


device, BATCH_SIZE = detectar_dispositivo()
print(f"🚀 Dispositivo activo : {device}")
print(f"📦 Batch size         : {BATCH_SIZE}\n")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\sáéíóúüñ]', ' ', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def cargar_textos_completo():
    archivos = [f for f in os.listdir(CORPUS_DIR) if f.endswith('.txt')]
    if not archivos:
        print("❌ No hay archivos .txt en data/corpus/")
        return ""
    
    todo_el_texto = []
    print(f"📚 Cargando {len(archivos)} archivos...")
    for archivo in archivos:
        ruta = os.path.join(CORPUS_DIR, archivo)
        print(f"   📄 {archivo}")
        with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
            texto = f.read()
        texto_limpio = limpiar_texto(texto)
        todo_el_texto.append(texto_limpio)
    
    texto_completo = " ".join(todo_el_texto)
    print(f"   ✅ {len(texto_completo.split())} palabras cargadas")
    return texto_completo


def crear_tokenizador(texto):
    palabras = texto.split()
    contador = Counter(palabras)
    palabras_filtradas = [p for p in palabras if contador[p] >= MIN_COUNT]
    vocabulario = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] + sorted(set(palabras_filtradas))
    
    palabra_a_idx = {p: i for i, p in enumerate(vocabulario)}
    idx_a_palabra = {i: p for p, i in palabra_a_idx.items()}
    
    print(f"   📖 Vocabulario: {len(vocabulario)} tokens")
    return vocabulario, palabra_a_idx, idx_a_palabra


def texto_a_tokens(texto, palabra_a_idx, block_size):
    palabras = texto.split()
    tokens = [palabra_a_idx.get(p, palabra_a_idx[UNK_TOKEN]) for p in palabras]

    # 🛡️ CORREGIDO: si el corpus tiene menos tokens que block_size, range()
    # no generaba nada y secuencias quedaba vacío. Eso después rompía el
    # entrenamiento con ZeroDivisionError (total_loss / n_batches, n_batches=0).
    if len(tokens) <= block_size:
        print(f"   ⚠️ El corpus tiene {len(tokens)} tokens, "
              f"menos que block_size ({block_size}). No se pueden generar secuencias.")
        return []

    secuencias = []
    for i in range(0, len(tokens) - block_size, block_size // 2):
        secuencia = tokens[i:i + block_size + 1]
        if len(secuencia) == block_size + 1:
            secuencias.append((secuencia[:-1], secuencia[1:]))
    
    return secuencias


class TextDataset(Dataset):
    def __init__(self, secuencias):
        self.entradas = torch.tensor([s[0] for s in secuencias], dtype=torch.long)
        self.salidas = torch.tensor([s[1] for s in secuencias], dtype=torch.long)
    
    def __len__(self):
        return len(self.entradas)
    
    def __getitem__(self, idx):
        return self.entradas[idx], self.salidas[idx]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, mask=None):
        # 🛡️ CORREGIDO: Ahora se pasa la máscara causal 'mask' a la atención
        # is_causal=True optimiza internamente el proceso en versiones modernas de PyTorch
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, is_causal=mask is not None)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(block_size, embed_dim)
        
        # Guardamos los bloques en una lista normal de módulos para iterar bien con la máscara
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size
        
        nn.init.xavier_uniform_(self.token_embed.weight)
        nn.init.xavier_uniform_(self.position_embed.weight)
    
    def forward(self, x):
        batch, seq_len = x.shape
        
        # 🛡️ CORREGIDO: Generar máscara causal triangular superior
        # Previene que el token 't' mire elementos en posiciones mayores a 't'
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        token_emb = self.token_embed(x)
        pos_emb = self.position_embed(pos)
        x = token_emb + pos_emb
        
        # Pasar los datos y la máscara bloque por bloque
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
    
    def generar(self, tokens_iniciales, max_tokens=100, temperatura=0.8):
        modo_previo_entrenamiento = self.training
        self.eval()
        tokens = tokens_iniciales.copy()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if len(tokens) > self.block_size:
                    contexto = tokens[-self.block_size:]
                else:
                    contexto = tokens
                
                # Mover el tensor temporal al dispositivo correcto (GPU/CPU)
                x = torch.tensor([contexto], dtype=torch.long, device=device)
                logits = self.forward(x)
                logits = logits[0, -1, :] / temperatura
                
                probs = torch.softmax(logits, dim=-1)
                siguiente = torch.multinomial(probs, 1).item()
                
                tokens.append(siguiente)
                if siguiente == 3:  # EOS_TOKEN
                    break

        # 🛡️ CORREGIDO: restaurar el modo de entrenamiento previo, para no dejar
        # el modelo "congelado" en eval() si se sigue entrenando después de generar.
        if modo_previo_entrenamiento:
            self.train()
        return tokens


def cargar_word2vec_pesos(vocab_size, embed_dim, palabra_a_idx):
    if not os.path.exists(W2V_PATH):
        print("   ⚠️ Word2Vec no encontrado en la ruta, iniciando con pesos aleatorios.")
        return None
    
    print(f"\n🧬 Cargando Word2Vec pre-entrenado...")
    checkpoint = torch.load(W2V_PATH, map_location=device)
    w2v_weights = checkpoint['model_state_dict']['in_embed.weight']
    
    with open("data/vocabulario_pytorch.json", 'r', encoding='utf-8') as f:
        w2v_data = json.load(f)
    w2v_vocab = w2v_data['palabra_a_idx']
    
    pesos_iniciales = torch.zeros(vocab_size, embed_dim)
    nn.init.xavier_uniform_(pesos_iniciales)
    
    palabras_cargadas = 0
    w2v_dim = w2v_weights.shape[1]
    
    for palabra, idx_transformer in palabra_a_idx.items():
        if palabra in w2v_vocab:
            idx_w2v = w2v_vocab[palabra]
            # Copiar dimensiones disponibles limitando al tamaño máximo
            dim_a_copiar = min(embed_dim, w2v_dim)
            pesos_iniciales[idx_transformer, :dim_a_copiar] = w2v_weights[idx_w2v, :dim_a_copiar]
            palabras_cargadas += 1
    
    print(f"   ✅ {palabras_cargadas}/{len(palabra_a_idx)} palabras heredadas con éxito.")
    return pesos_iniciales


def entrenar_modelo(secuencias, vocab_size, palabra_a_idx, modelo_previo=None):
    # 🛡️ CORREGIDO: validar que haya secuencias antes de crear el DataLoader.
    # Si llega vacío (corpus muy chico), el loop de entrenamiento itera cero
    # veces y "total_loss / n_batches" revienta con ZeroDivisionError.
    if not secuencias:
        raise ValueError(
            "No se generaron secuencias de entrenamiento. "
            "El corpus es demasiado corto para BLOCK_SIZE="
            f"{BLOCK_SIZE}, o todas las palabras fueron filtradas por MIN_COUNT="
            f"{MIN_COUNT}. Agrega más texto a data/corpus/ o reduce BLOCK_SIZE/MIN_COUNT."
        )

    dataset = TextDataset(secuencias)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 🛡️ CORREGIDO: Controlar la reanudación antes de aplicar cualquier inyección de pesos
    if modelo_previo is not None:
        modelo = modelo_previo
        print(f"\n🔄 REANUDANDO entrenamiento desde el estado previo...")
    else:
        modelo = MiniGPT(vocab_size, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE)
        print(f"\n🆕 INICIANDO modelo desde cero...")
        
        # Inyectar Word2Vec solo si es un modelo nuevo
        pesos_w2v = cargar_word2vec_pesos(vocab_size, EMBEDDING_DIM, palabra_a_idx)
        if pesos_w2v is not None:
            modelo.token_embed.weight.data = pesos_w2v.to(device)
            print(f"   🧬 ¡Transformer inicializado con la semántica del Word2Vec!")

    # Mover el modelo completo al dispositivo (GPU o CPU)
    modelo = modelo.to(device)
    
    total_params = sum(p.numel() for p in modelo.parameters())
    print(f"   Parámetros totales: {total_params:,} (~{total_params/1e6:.1f}M)")
    
    optimizer = optim.AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignora PAD_TOKEN
    
    print(f"   Épocas: {EPOCHS} | Batch: {BATCH_SIZE} | Dim Embedding: {EMBEDDING_DIM}")
    
    modelo.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        n_batches = 0
        
        for entradas, salidas in dataloader:
            # 🛡️ CORREGIDO: Enviar los lotes de datos al dispositivo (GPU/CPU)
            entradas, salidas = entradas.to(device), salidas.to(device)
            
            logits = modelo(entradas)
            loss = criterion(logits.view(-1, vocab_size), salidas.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if n_batches % 100 == 0:
                print(f"      Batch {n_batches} | Loss actual: {loss.item():.4f}")
        
        avg_loss = total_loss / n_batches
        print(f"   ✨ Época {epoch+1}/{EPOCHS} Completada | Loss Promedio: {avg_loss:.4f}")
    
    return modelo


def guardar_modelo(modelo, vocabulario, palabra_a_idx, idx_a_palabra, total_epochs=0):
    # Guardamos siempre moviendo los pesos a CPU por portabilidad
    torch.save({
        'model_state_dict': modelo.cpu().state_dict(),
        'vocab_size': len(vocabulario),
        'embed_dim': EMBEDDING_DIM,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'block_size': BLOCK_SIZE,
        'epochs_entrenados': total_epochs,
    }, MODEL_PATH)
    print(f"\n💾 Modelo guardado en {MODEL_PATH}")
    
    # Devolver el modelo a su dispositivo original tras guardarlo
    modelo.to(device)
    
    data = {
        "vocabulario": vocabulario,
        "palabra_a_idx": palabra_a_idx,
        "idx_a_palabra": {str(k): v for k, v in idx_a_palabra.items()},
        "config": {
            "embed_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "block_size": BLOCK_SIZE
        }
    }
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"💾 Vocabulario guardado en {VOCAB_PATH}")


def probar_modelo(modelo, palabra_a_idx, idx_a_palabra):
    print("\n🔮 Probando generación de texto autorregresiva:")
    pruebas = [
        "la guerra es",
        "el tanque alemán",
        "python es un lenguaje",
        "la inteligencia artificial",
    ]
    
    for prompt in pruebas:
        tokens = [palabra_a_idx.get(p, palabra_a_idx[UNK_TOKEN]) for p in prompt.split()]
        generados = modelo.generar(tokens, max_tokens=25, temperatura=0.7)
        texto = " ".join([idx_a_palabra.get(t, "?") for t in generados])
        print(f"\n   Prompt: '{prompt}'")
        print(f"   Generado: '{texto}'")


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("   🧠 TRANSFORMER CAUSAL REAL - MINIGPT")
    print("=" * 60)
    
    texto = cargar_textos_completo()
    if not texto:
        exit()
    
    vocabulario, palabra_a_idx, idx_a_palabra = crear_tokenizador(texto)
    secuencias = texto_a_tokens(texto, palabra_a_idx, BLOCK_SIZE)
    print(f"   🔗 {len(secuencias)} secuencias de entrenamiento listas.")
    
    modelo = entrenar_modelo(secuencias, len(vocabulario), palabra_a_idx)
    guardar_modelo(modelo, vocabulario, palabra_a_idx, idx_a_palabra, EPOCHS)
    probar_modelo(modelo, palabra_a_idx, idx_a_palabra)
    
    print(f"\n✅ ¡Proceso terminado con éxito!")