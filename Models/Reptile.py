import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
import wandb
from Utils.globals import *
from Utils.utils import *

# ==========================================
# 0. HELPER PARA MEDICIÓN DE TIEMPO
# ==========================================
class CudaTimer:
    """Mide tiempo real en GPU usando eventos de CUDA."""
    def __init__(self, device):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed = 0.0

    def start(self):
        if self.device.type == 'cuda':
            self.start_event.record()
        else:
            self.start_time = time.time()

    def stop(self):
        if self.device.type == 'cuda':
            self.end_event.record()
            torch.cuda.synchronize() 
            self.elapsed = self.start_event.elapsed_time(self.end_event) # devuelve milisegundos
        else:
            self.elapsed = (time.time() - self.start_time) * 1000 # a ms

    def get_time_ms(self):
        return self.elapsed

def train_epoch(meta_model, train_loader, val_loader, n_way, k_shot, q_query, inner_lr, inner_steps, epsilon, device):
    """
    Trains the meta_model for one epoch (e.g., 100 episodes) using manual tensor reshaping.
    """
    meta_model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    # Iterate over the raw DataLoader batches
    for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = images.to(device)
        
        # --- 1. MANUALLY PREPARE DATA ---
        # Input shape: [Batch_Size, C, H, W] where Batch_Size = n_way * (K_shot + Q_query)
        # Reshape to: [n_way, K+Q, C, H, W]
        data = images.view(n_way, k_shot + q_query, 3, 84, 84)

        # Split Support and Query
        # x_support: [n_way * K_shot, C, H, W]
        x_support = data[:, :k_shot].contiguous().view(-1, 3, 84, 84)
        y_support = torch.arange(n_way).repeat_interleave(k_shot).to(device)
        
        # x_query: [n_way * Q_query, C, H, W]
        x_query   = data[:, k_shot:].contiguous().view(-1, 3, 84, 84)
        y_query   = torch.arange(n_way).repeat_interleave(q_query).to(device)

        # --- 2. INNER LOOP (Fine-Tune a clone) ---
        fast_model = copy.deepcopy(meta_model)
        fast_model.train()
        inner_optimizer = optim.SGD(fast_model.parameters(), lr=inner_lr)
        
        # Train the clone on the Support Set
        for _ in range(inner_steps):
            logits = fast_model(x_support)
            loss = criterion(logits, y_support)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
            
        # --- 3. REPTILE UPDATE (Meta-Model Soft Update) ---
        # Update meta-weights towards fast-weights
        # Formula: meta_weight = meta_weight + epsilon * (fast_weight - meta_weight)
        with torch.no_grad():
            for meta_param, fast_param in zip(meta_model.parameters(), fast_model.parameters()):
                meta_param.data.add_(fast_param.data - meta_param.data, alpha=epsilon)

        # --- 4. LOGGING ---
        # Calculate loss on query set just for progress tracking (no backward pass)
        with torch.no_grad():
            q_logits = fast_model(x_query)
            q_loss = criterion(q_logits, y_query)
            total_loss += q_loss.item()

    return total_loss / len(train_loader)

def evaluate(meta_model, test_loader, n_way, k_shot, q_query, inner_lr, inner_steps, device):
    """
    Evaluates the meta_model on the test set using manual tensor reshaping.
    """
    meta_model.eval()
    criterion = nn.CrossEntropyLoss()
    total_acc = 0
    
    adapt_times_ms = [] # List to store adaptation + inference time per episode
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(test_loader, desc="Evaluating", leave=False)):
            images = images.to(device)
            
            # --- 1. MANUALLY PREPARE DATA ---
            data = images.view(n_way, k_shot + q_query, 3, 84, 84)

            x_support = data[:, :k_shot].contiguous().view(-1, 3, 84, 84)
            y_support = torch.arange(n_way).repeat_interleave(k_shot).to(device)

            # Apply data augmentation ONLY to support set
            x_support = apply_support_aug(x_support)
            
            x_query   = data[:, k_shot:].contiguous().view(-1, 3, 84, 84)
            y_query   = torch.arange(n_way).repeat_interleave(q_query).to(device)
            
             # --- MEASURE TIME (ADAPTATION + INFERENCE) ---
            # Esto mide: Generar Pesos (Adaptación) + Forward Query (Inferencia)
            timer = CudaTimer(device)
            timer.start()
            
            # --- 2. CLONE & FINE-TUNE ---
            fast_model = copy.deepcopy(meta_model)
            fast_model.train() # The clone must be in train mode to update weights
            inner_optimizer = optim.SGD(fast_model.parameters(), lr=inner_lr)
            
            # ENABLE GRADIENTS explicitly for the inner loop
            with torch.enable_grad():
                for _ in range(inner_steps):
                    logits = fast_model(x_support)
                    loss = criterion(logits, y_support)
                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()
            
            # --- 3. EVALUATE ACCURACY ---
            fast_model.eval()
            logits = fast_model(x_query)
            preds = logits.argmax(dim=1)
            correct = (preds == y_query).sum().item()
            total = y_query.size(0)
            
            timer.stop()
            total_acc += correct / total
            adapt_times_ms.append(timer.get_time_ms())
            mean_time = np.mean(adapt_times_ms)
                

    return total_acc / len(test_loader), mean_time

def train_reptile(meta_model, train_loader, test_loader, val_loader, device, n_way, n_shot, n_query):
    wandb.login(key="93d025aa0577b011c6d4081b9d4dc7daeb60ee6b")
    wandb.init(
        project=f"Pokemon-Reptile-Oak", 
        name=f"reptile-random-{n_way}-{n_shot}",   
        config={
            "n_way": n_way,
            "n_shot": n_shot,
            "n_query": n_query,
            "inner_lr": INNER_LR,
            "inner_steps": INNER_STEPS,
            "epsilon": EPSILON,
            "max_epochs": MAX_EPOCHS,
            "episodes_per_epoch": EPISODES_PER_EPOCH,
        }
    )
    print(f"Starting Training: {MAX_EPOCHS} Epochs x {EPISODES_PER_EPOCH} Episodes = {MAX_EPOCHS*EPISODES_PER_EPOCH} Total Episodes")
    
    best_val_acc = 0.0
    total_train_time = 0.0
    timer_global = CudaTimer(device) # Timer para medir tiempo total de entrenamiento
    for epoch in range(1, MAX_EPOCHS + 1):
        timer_global.start()
        # A. TRAIN ONE EPOCH
        avg_loss = train_epoch(
            meta_model, train_loader, val_loader,
            n_way, n_shot, n_query, 
            INNER_LR, INNER_STEPS, EPSILON, device
        )
        
        timer_global.stop()
        epoch_duration_sec = timer_global.get_time_sec()
        total_train_time += epoch_duration_sec
        wandb.log({
        "train/loss": avg_loss, 
        "epoch": epoch, 
        "time_per_epoch_sec": epoch_duration_sec, 
        "total_train_time_min": total_train_time / 60.0})
         
        # B. VALIDATE EVERY X EPOCHS
        if epoch % 5 == 0:
            val_acc, _ = evaluate(
                meta_model, val_loader, 
                n_way, n_shot, n_query, 
                INNER_LR, INNER_STEPS, device
            )

            wandb.log({"val/accuracy": val_acc, "epoch": epoch})
            print(f"Epoch {epoch}: Train Loss {avg_loss:.4f} | Val Acc {val_acc*100:.2f}%")
            
            # Save Best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(meta_model.state_dict(), f"Results/Models_pth/Reptile_pth/reptile-gen-{n_way}-{n_shot}-da.pth")
                print(" -> New Best Model Saved!")
        else:
            print(f"Epoch {epoch}: Train Loss {avg_loss:.4f}")

    print("\n--- Training Finished. Running Final Test ---")
    meta_model.load_state_dict(torch.load(f"Results/Models_pth/Reptile_pth/reptile-gen-{n_way}-{n_shot}-da.pth"))
    
    test_acc, mean_time = evaluate(
        meta_model, test_loader, 
        n_way, n_shot, n_query, 
        INNER_LR, INNER_STEPS, device
    )
    wandb.log({"test/accuracy": test_acc})
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    wandb.log({"test/mean_adaptation_time_ms": mean_time})
    print(f"Mean Adaptation + Inference Time per Episode: {mean_time:.8f} milseconds")

    wandb.finish()