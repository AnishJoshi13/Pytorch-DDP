import os
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, DistributedSampler

######################
# 1) Define the model
######################
class SimpleNet(nn.Module):
    def __init__(self, in_dim=3 * 32 * 32, hidden=1000, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_inner = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # Flatten input [B, 3, 32, 32] -> [B, 3*32*32]
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        for _ in range(20):
            x = self.fc_inner(x)
            x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


##############################
# 2) Training loop (per rank)
##############################
def ddp_train_loop(local_rank, world_size, args):
    """
    local_rank: index in [0..world_size-1], assigned by mp.spawn
    world_size: total number of GPU processes
    args: dictionary with config (backend, epochs, save_path, etc.)
    """

    # -- a) Create a FileStore and init process group --
    store_path = os.path.join(args["save_path"], "filestore")
    os.makedirs(store_path, exist_ok=True)

    # We create a FileStore. The second argument is the number of processes (world_size).
    store = dist.FileStore(os.path.join(store_path, "store"), world_size)

    dist.init_process_group(
        backend=args["backend"],   # e.g. "gloo" or "nccl" if on Linux with GPUs
        store=store,
        rank=local_rank,
        world_size=world_size
    )

    # -- b) Create local model & DDP wrapper --
    # Map local_rank -> GPU device (assumes at least `world_size` GPUs)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    ddp_model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    # -- c) Create dataset / dataloader --
    train_set = CIFAR10(
        root=args["data_dir"],
        train=True,
        download=True,
        transform=ToTensor()
    )
    sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=64,
        sampler=sampler,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True
    )

    # -- d) Setup optimizer, criterion --
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # -- e) Training --
    for epoch in range(args["epochs"]):
        sampler.set_epoch(epoch)  # ensures a different shuffle each epoch
        ddp_model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print only on rank=0 (the "main" GPU process).
            if batch_idx % 100 == 0 and local_rank == 0:
                print(f"[GPU Rank {local_rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # If local_rank=0, save checkpoint
        if local_rank == 0:
            state_dict = model.state_dict()
            ckpt_path = os.path.join(args["save_path"], f"model_epoch{epoch}.pt")
            torch.save(state_dict, ckpt_path)
            print(f"[GPU Rank 0] Saved {ckpt_path}")

    dist.destroy_process_group()
    print(f"[GPU Rank {local_rank}] training complete.")


##########################
# 3) CPU test process
##########################
def cpu_test_proc(args):
    """
    Single CPU process that checks for each epoch exactly once,
    then exits when done.
    """
    print("[Test Process] Started, using CPU")

    test_set = CIFAR10(
        root=args["data_dir"],
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_loader = DataLoader(
        test_set,
        batch_size=10000,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args["epochs"]):
        ckpt_path = os.path.join(args["save_path"], f"model_epoch{epoch}.pt")

        # Wait until the checkpoint for this epoch appears
        while not os.path.exists(ckpt_path):
            time.sleep(5)

        # Load and evaluate
        model = SimpleNet()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        print(f"[Test Process: {ckpt_path}] Epoch {epoch}, "
              f"Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}")

    print("[Test Process] All epochs tested. Exiting.")


##########################
# 4) Main entry point
##########################
def main():
    # Remove old checkpoints if desired
    for f in glob.glob('./checkpoints/*'):
        print(f"removing {f}")
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            print(f"Error removing {f}: {e}")

    args = {
        "backend": "gloo",   # or "nccl" on Linux with multi-GPU
        "epochs": 2,
        "data_dir": "./data",
        "save_path": "./checkpoints"
    }
    os.makedirs(args["save_path"], exist_ok=True)

    # For example, 2 GPU processes => world_size=2
    # Adjust for however many GPUs you have
    world_size = 1

    # 1) Start the single CPU test process
    test_process = mp.Process(target=cpu_test_proc, args=(args,))
    test_process.start()

    # 2) Spawn multiple GPU processes
    mp.spawn(
        fn=ddp_train_loop,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

    # 3) After GPU training finishes, wait for CPU test process to finish
    test_process.join()
    print("[Main] All processes finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
