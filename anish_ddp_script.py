import os
import time
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
    def __init__(self, in_dim=3 * 32 * 32, hidden=100, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # Flatten input [B, 3, 32, 32] -> [B, 3*32*32]
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


##############################
# 2) Training loop (per rank)
##############################
def ddp_train_loop(rank, world_size, args):
    """
    rank: process rank [0..world_size-1]
    world_size: total number of DDP processes
    args: dictionary with config (backend, epochs, save_path, etc.)
    """
    # a) Create a FileStore and init PG (no TCP or NCCL)
    store_path = os.path.join(args["save_path"], "filestore")
    os.makedirs(store_path, exist_ok=True)

    # The second argument to FileStore is the number of processes (world_size)
    store = dist.FileStore(os.path.join(store_path, "store"), world_size)

    dist.init_process_group(
        backend=args["backend"],  # "gloo"
        store=store,
        world_size=world_size,
        rank=rank
    )

    # b) Create local model & DDP wrapper
    model = SimpleNet()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ddp_model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    # c) Create dataset / dataloader
    train_set = CIFAR10(root=args["data_dir"], train=True, download=True, transform=ToTensor())
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=64, sampler=sampler, num_workers=2)

    # d) Setup optimizer, criterion
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # e) Training
    for epoch in range(args["epochs"]):
        sampler.set_epoch(epoch)
        ddp_model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print(f"[Rank {rank}] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # f) If rank=0, save checkpoint
        if rank == 0:
            state_dict = model.state_dict()
            ckpt_path = os.path.join(args["save_path"], f"model_epoch{epoch}.pt")
            torch.save(state_dict, ckpt_path)
            print(f"[Rank 0] Saved {ckpt_path}")

    dist.destroy_process_group()
    print(f"[Rank {rank}] training complete.")


##########################
# 3) CPU test process
##########################
def cpu_test_proc(args):
    """
    Runs on CPU and checks for each epoch exactly once,
    then exits when done (so we don't get stuck waiting).
    """
    print("[Test Process] Started, using CPU")
    test_set = CIFAR10(root=args["data_dir"], train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    # We'll do exactly 'args["epochs"]' test checks
    for epoch in range(args["epochs"]):
        ckpt_path = os.path.join(args["save_path"], f"model_epoch{epoch}.pt")

        # Wait until the checkpoint for this epoch appears
        while not os.path.exists(ckpt_path):
            time.sleep(5)  # Keep checking every 5 seconds

        # Once found, load and evaluate
        model = SimpleNet()
        # FutureWarning: if you want to avoid full pickle loading, use 'weights_only=True'
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
        print(f"[Test Process] Epoch {epoch}, Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}")

    print("[Test Process] All epochs tested. Exiting.")


##########################
# 4) Main entry point
##########################
def main():
    args = {
        # Force "gloo" because "nccl" is not compiled on Windows
        "backend": "gloo",
        "epochs": 2,
        "data_dir": "./data",
        "save_path": "./checkpoints"
    }
    os.makedirs(args["save_path"], exist_ok=True)

    world_size = 1  # Adjust if you have more GPUs

    # Remove daemon=True so the test process can spawn DataLoader workers
    test_process = mp.Process(target=cpu_test_proc, args=(args,))
    test_process.start()

    mp.spawn(
        fn=ddp_train_loop,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

    test_process.join()
    print("[Main] All processes finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
