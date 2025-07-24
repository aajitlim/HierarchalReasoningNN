import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

# --- The Worker with the CORRECTED Hierarchical Memory ---
class HierarchicalMemoryWorker(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_sectors, memories_per_sector, key_dim, value_dim, top_k):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sectors = num_sectors
        self.memories_per_sector = memories_per_sector
        self.k = top_k

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 1. The Master Router
        self.sector_keys = nn.Parameter(torch.randn(num_sectors, embed_dim))
        nn.init.xavier_uniform_(self.sector_keys)

        # --- THE FIX ---
        # 2. The Hierarchical Memory Bank
        # self.memory_sectors must be an nn.ModuleList to properly register the modules within it.
        self.memory_sectors = nn.ModuleList()
        for _ in range(num_sectors):
            # Each 'sector' is a ParameterDict, which is the correct container for named parameters.
            # This correctly resolves the "torch.FloatTensor is not a Module subclass" error.
            sector = nn.ParameterDict({
                'knowledge_matrix': nn.Parameter(torch.randn(memories_per_sector, key_dim, value_dim)),
                'memory_keys': nn.Parameter(torch.randn(memories_per_sector, embed_dim)),
            })
            nn.init.xavier_uniform_(sector.knowledge_matrix)
            nn.init.xavier_uniform_(sector.memory_keys)
            self.memory_sectors.append(sector)
            
        # Shared processing layers
        self.query_proj = nn.Linear(embed_dim, key_dim)
        self.output_proj = nn.Linear(value_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, guidance_signal=None):
        x_emb = self.token_embedding(x)
        
        # --- Master Routing Decision ---
        sector_scores = torch.matmul(x_emb, self.sector_keys.t())
        if guidance_signal is not None:
            sector_scores = sector_scores + guidance_signal
        sector_distribution = F.softmax(sector_scores, dim=-1)
        
        # --- Parallel Local Lookups within each Sector ---
        all_sector_outputs = []
        for sector in self.memory_sectors:
            local_lookup_scores = torch.matmul(x_emb, sector.memory_keys.t())
            local_dist = F.softmax(local_lookup_scores, dim=-1)
            
            top_k_weights, top_k_indices = torch.topk(local_dist, self.k, dim=-1)
            if (top_k_weights.sum(dim=-1, keepdim=True) > 1e-9).all():
                top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

            retrieved_transformations = sector.knowledge_matrix[top_k_indices]
            token_query = self.query_proj(x_emb)
            transformed_vectors = torch.matmul(token_query.unsqueeze(2).unsqueeze(3), retrieved_transformations).squeeze(3)
            contextualized_vector = torch.sum(transformed_vectors * top_k_weights.unsqueeze(-1), dim=2)
            all_sector_outputs.append(contextualized_vector)

        stacked_sector_outputs = torch.stack(all_sector_outputs).permute(1, 2, 0, 3)

        # --- Final Aggregation ---
        final_context_vector = torch.sum(stacked_sector_outputs * sector_distribution.unsqueeze(-1), dim=2)
        projected_output = self.output_proj(final_context_vector)
        output = self.norm(x_emb + projected_output)
        
        return output, sector_distribution

# --- The Supervisor and Classifier (Unchanged) ---
class SupervisorLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_sectors):
        super().__init__()
        self.meta_attention = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*2)
        self.recommender = nn.Linear(embed_dim, num_sectors)
    def forward(self, worker_output):
        gestalt = self.meta_attention(worker_output).mean(dim=1)
        return self.recommender(gestalt)

class HierarchicalClassifier(nn.Module):
    def __init__(self, worker, supervisor, num_classes):
        super().__init__()
        self.worker = worker; self.supervisor = supervisor
        self.output = nn.Linear(worker.embed_dim, num_classes)
    def forward(self, x):
        worker_output_pass1, _ = self.worker(x)
        guidance_signal = self.supervisor(worker_output_pass1)
        worker_output_pass2, sector_dist = self.worker(x, guidance_signal.unsqueeze(1))
        final_representation = worker_output_pass2.mean(dim=1)
        final_logits = self.output(final_representation)
        return final_logits, sector_dist

# --- Data Generation (Unchanged) ---
def generate_harder_task_data(num_samples, seq_len, vocab_size, num_contexts, num_keywords, num_modifiers):
    filler_end=11; ctx_start=filler_end; ctx_end=ctx_start+num_contexts
    key_start=ctx_end; key_end=key_start+num_keywords; mod_start=key_end
    mod_end=mod_start+num_modifiers; contexts=np.arange(ctx_start,ctx_end)
    keywords=np.arange(key_start,key_end); modifiers=np.arange(mod_start,mod_end)
    fillers=np.arange(1,filler_end); vocab_size=max(vocab_size,mod_end)
    X=np.zeros((num_samples,seq_len),dtype=np.int64); y=np.zeros(num_samples,dtype=np.int64)
    num_classes=num_contexts*num_keywords*num_modifiers; class_map={}; i=0
    for c in contexts:
      for k in keywords:
        for m in modifiers:
          class_map[(c,k,m)] = i; i+=1
    for i in range(num_samples):
      context_word=np.random.choice(contexts); keyword=np.random.choice(keywords)
      modifier=np.random.choice(modifiers); y[i]=class_map[(context_word,keyword,modifier)]
      sentence=np.random.choice(fillers,seq_len); pos=np.random.choice(range(seq_len),3,replace=False)
      sentence[pos[0]],sentence[pos[1]],sentence[pos[2]]=context_word,keyword,modifier
      X[i,:]=sentence
    return torch.from_numpy(X),torch.from_numpy(y),vocab_size,num_classes

# --- Main Scaling Experiment Script (as you provided) ---
if __name__ == "__main__":
    # Corrected DEVICE handling for single-GPU or multi-GPU setups
    if torch.cuda.is_available():
        try:
            DEVICE = "cuda:1"
            torch.cuda.set_device(DEVICE) # Attempt to set to cuda:1
        except (RuntimeError, AssertionError):
            print("cuda:1 not available, falling back to cuda:0")
            DEVICE = "cuda:0"
    else:
        DEVICE = "cpu"
    print(f"--- Using device: {DEVICE} ---")

    # We will test the original scaling levels that failed
    scaling_levels = [16]
    results = {}
    
    epochs = 20
    eval_interval = 2
    
    for level in scaling_levels:
        num_contexts = level
        num_keywords = level
        num_modifiers = level
        num_classes = num_contexts**3
        
        experiment_label = f"{num_classes} classes ({level}^3)"
        print(f"\n--- Starting Experiment: {experiment_label} ---")
        results[experiment_label] = {'epochs': [], 'accuracy': []}
        
        # Using the dynamic hyperparameters from our discussion
        base_embed_dim = 32
        embed_dim = base_embed_dim + (level * 4)
        memories_per_sector = 16
        base_key_dim = 16
        base_value_dim = 32
        key_dim = base_key_dim + (level * 2)
        value_dim = base_value_dim + (level * 4)
        num_sectors = 16
        top_k = 3
        num_heads = 4
        
        print(f"Scaled Hyperparameters: embed_dim={embed_dim}, memories_per_sector={memories_per_sector}, key_dim={key_dim}, value_dim={value_dim}")

        X_train, y_train, vocab_size, _ = generate_harder_task_data(15000, 25, 200, num_contexts, num_keywords, num_modifiers)
        X_test, y_test, _, _ = generate_harder_task_data(2000, 25, vocab_size, num_contexts, num_keywords, num_modifiers)
        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

        worker = HierarchicalMemoryWorker(vocab_size, embed_dim, num_sectors, memories_per_sector, key_dim, value_dim, top_k).to(DEVICE)
        supervisor = SupervisorLayer(embed_dim, num_heads, num_sectors).to(DEVICE)
        model = HierarchicalClassifier(worker, supervisor, num_classes).to(DEVICE)
        
        lr = 0.002 - (level * 0.00002)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        batch_size = 128
        
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(range(0, X_train.size(0), batch_size), desc=f"Epoch {epoch+1}/{epochs} | Loss: N/A")
            permutation = torch.randperm(X_train.size(0))
            for i in pbar:
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                outputs, _ = model(X_train[indices])
                loss = criterion(outputs, y_train[indices])
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch+1}/{epochs} | Batch Loss: {loss.item():.4f}")
            
            if (epoch + 1) % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    all_test_outputs = []
                    for i in range(0, X_test.size(0), batch_size):
                        test_outputs, _ = model(X_test[i:i+batch_size])
                        all_test_outputs.append(test_outputs)
                    all_test_outputs = torch.cat(all_test_outputs, dim=0)
                    _, predicted = torch.max(all_test_outputs, 1)
                    accuracy = (predicted == y_test).float().mean().item()
                    results[experiment_label]['epochs'].append(epoch + 1)
                    results[experiment_label]['accuracy'].append(accuracy)
                    tqdm.write(f"--- Eval @ Epoch {epoch+1} for {experiment_label}: Test Accuracy: {accuracy:.4f} ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(scaling_levels)))
    for i, (label, data) in enumerate(results.items()):
        plt.plot(data['epochs'], data['accuracy'], marker='o', linestyle='-', color=colors[i], label=label)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Hierarchical Model with DYNAMIC Capacity vs. Task Complexity', fontsize=16)
    plt.legend(title='Experiment Scale', fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.xticks(np.arange(0, epochs + 1, eval_interval))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()