import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, DistributedSampler

#################
# Simple Model
#################
class SimpleNet(nn.Module):
    def __init__(self, in_dim=3 * 32 * 32, hidden=1000, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_inner = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        for _ in range(5):
            x = self.fc_inner(x)
            x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


def run_cpu_test(epochs=2, ckpt_dir="./checkpoints"):
    """
    Rank=0: CPU test logic. Wait for model_epoch{e}.pt each epoch, load & test on CPU.
    """
    print("[CPU Rank 0] Test process started.")
    test_set = CIFAR10(root="./data", train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        ckpt_path = f"{ckpt_dir}/model_epoch{epoch}.pt"
        while not os.path.exists(ckpt_path):
            time.sleep(5)  # wait for GPU rank to save checkpoint

        cpu_model = SimpleNet()
        cpu_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        cpu_model.eval()

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = cpu_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        print(f"[CPU Rank 0] Epoch={epoch}, Accuracy={accuracy:.2f}%, Loss={avg_loss:.4f}")

    print("[CPU Rank 0] Test done.")


def run_gpu_train(rank, world_size, epochs=2, ckpt_dir="./checkpoints"):
    """
    Ranks 1..N run GPU training. Rank=0 is CPU test, so skip training there.
    We'll use a FileStore to avoid libuv-based TCPStore.
    """
    # 1) Create a local folder for the FileStore
    store_dir = "./filestore"
    os.makedirs(store_dir, exist_ok=True)

    # 2) Build the FileStore for all ranks to share
    #    The second argument is the total # of processes: 1 CPU + (N GPUs) = world_size
    store = dist.FileStore(os.path.join(store_dir, "store"), world_size)

    # 3) Init process group with "gloo" or "nccl" (on Linux). On Windows, "gloo" is safer.
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        # CPU side does no training here, just part of the group so it can barrier() if needed.
        print("[CPU Rank 0 inside run_gpu_train] No training logic.")
        dist.barrier()
        dist.destroy_process_group()
        return

    # If rank >= 1 => GPU training
    gpu_index = rank - 1  # rank=1 -> cuda:0, rank=2 -> cuda:1, ...
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = SimpleNet().to(device)
    from torch.nn.parallel import DistributedDataParallel as DDP
    ddp_model = DDP(model, device_ids=[device])

    train_set = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    # We have (world_size-1) GPU ranks total. For sampler rank=1 => index=0, etc.
    sampler = DistributedSampler(
        dataset=train_set,
        num_replicas=(world_size - 1),
        rank=(rank - 1),
        shuffle=True
    )
    train_loader = DataLoader(train_set, batch_size=64, sampler=sampler, num_workers=2)

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        sampler.set_epoch(ep)
        ddp_model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if rank == 1 and batch_idx % 100 == 0:
                print(f"[GPU Rank 1] epoch={ep}, batch={batch_idx}, loss={loss.item():.4f}")

        # rank=1 saves model after each epoch
        if rank == 1:
            torch.save(model.state_dict(), f"{ckpt_dir}/model_epoch{ep}.pt")
            print(f"[GPU Rank 1] saved model_epoch{ep}.pt")

    dist.barrier()
    dist.destroy_process_group()
    print(f"[GPU Rank {rank}] training complete.")


def main():
    # 1) Parse environment variables from torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    EPOCHS = 2
    CKPT_DIR = "./checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Decide if rank=0 => CPU test, else GPU
    if rank == 0:
        run_cpu_test(epochs=EPOCHS, ckpt_dir=CKPT_DIR)
    else:
        run_gpu_train(rank=rank, world_size=world_size, epochs=EPOCHS, ckpt_dir=CKPT_DIR)


if __name__ == "__main__":
    main()
