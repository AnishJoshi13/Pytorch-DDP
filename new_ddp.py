import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, DistributedSampler

###############################
# 1) Parse environment vars
###############################
try:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
except KeyError:
    raise RuntimeError("Please launch via torchrun so that LOCAL_RANK & WORLD_SIZE are set.")


######################
# Define the model
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


############################################
# 2) GPU Worker Logic (local_rank >= 1)
############################################
def run_gpu_process(local_rank: int, world_size: int, epochs: int, save_path: str):
    """
    Each GPU process trains on its share of data,
    sends last-batch info to rank=0 (CPU).
    Receives feedback from CPU about accuracy,
    if desired.
    """
    # Initialize NCCL backend for multi-GPU comm
    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    # Set correct GPU device
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    # Create model & wrap in DDP
    model = SimpleNet().to(device)
    ddp_model = DDP(model, device_ids=[device], find_unused_parameters=False)

    # Create DataLoader
    train_set = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    sampler = DistributedSampler(train_set, num_replicas=world_size - 1, rank=local_rank - 1)
    # ^ Subtle note: if we have 1 CPU rank + (world_size - 1) GPU ranks,
    #   then the GPU ranks range from 1..(world_size-1).
    #   So, we do rank=local_rank-1 for sampling.
    train_loader = DataLoader(
        train_set,
        batch_size=64,
        sampler=sampler,
        num_workers=2,
        shuffle=False,
    )

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
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

            if batch_idx % 50 == 0:
                # 1) Send the last-batch GPU loss to CPU rank=0
                loss_tensor = torch.tensor([loss.item()], dtype=torch.float32).to("cpu")
                dist.send(tensor=loss_tensor, dst=0)  # send to rank=0

                # 2) Also send the last batch (images, labels) for CPU inference (demonstration)
                #    We'll send shapes first, then the data.
                cpu_images = images.cpu()
                shape_tensor = torch.tensor(cpu_images.shape, dtype=torch.long)  # e.g. (64, 3, 32, 32)
                dist.send(tensor=shape_tensor, dst=0)

                # Flatten the images so we can send as a 1D tensor
                flat_images = cpu_images.view(-1)
                dist.send(tensor=flat_images, dst=0)

                # Send labels
                shape_labels = torch.tensor(labels.shape, dtype=torch.long)
                dist.send(shape_labels, dst=0)
                dist.send(labels.cpu(), dst=0)

                # 3) Wait for CPU to send back an "accuracy" float for this batch
                acc_tensor = torch.zeros(1, dtype=torch.float32)
                dist.recv(acc_tensor, src=0)  # blocking receive from CPU
                print(
                    f"[GPU {local_rank}] epoch={epoch}, batch={batch_idx}, loss={loss.item():.4f}, CPU-acc={acc_tensor.item():.2f}")

        #### "Saving" the model is delegated to CPU.
        #### We'll broadcast our state_dict to CPU at the end of epoch.
        # 4) Gather model parameters on CPU
        #    We'll do a naive approach: rank=local_rank sends each param to CPU individually.
        #    If you have a large model, see PyTorch docs for more efficient ways.
        if local_rank == 1:
            # Just pick rank=1 to handle the final broadcast to CPU.
            # (Or you can do a gather for each GPU rank.)
            for name, param in model.named_parameters():
                p_data = param.data.cpu()
                shape = torch.tensor(p_data.shape, dtype=torch.long)
                dist.send(shape, dst=0)
                dist.send(p_data.flatten(), dst=0)

            # Tell CPU we're done sending
            done_signal = torch.tensor([-1], dtype=torch.long)
            dist.send(done_signal, dst=0)

    dist.destroy_process_group()
    print(f"[GPU {local_rank}] training complete.")


##################################################
# 3) CPU Controller / Test Logic (local_rank=0)
##################################################
def run_cpu_process(world_size: int, epochs: int, save_path: str):
    """
    Single CPU process that:
      1) Receives losses from all GPUs, prints them.
      2) Receives last-batch data from GPU, does inference with CPU model
         (just for demonstration), sends accuracy back to the GPU.
      3) Gathers final model parameters from rank=1 (or each GPU)
         and writes them to disk so GPU doesn’t handle I/O overhead.
      4) Also runs a normal test set evaluation each epoch (optional).
    """
    # Initialize a process group for rank=0 with the same "nccl" or "gloo"?
    # Actually "nccl" won't typically work on CPU-only, so let's do "gloo" just for rank=0.
    dist.init_process_group(backend="gloo", world_size=world_size, rank=0)

    print("[CPU 0] Starting controller process.")
    # Create a CPU model for inference
    cpu_model = SimpleNet()
    cpu_model.eval()

    # If we want to do a normal test loop each epoch, we’d do so here...
    # We'll keep it minimal.

    # We expect (world_size - 1) GPU ranks sending data.
    # Let's just do a loop for (epochs * #batches * ???).
    # Because we don't know exactly how many times the GPU will send us stuff,
    # we do a while True approach with a break condition.
    # For demonstration, we’ll assume total steps = epochs * 50, etc.

    steps_received = 0
    total_expected = epochs * 500  # e.g. if ~500 steps per epoch across all GPUs, a guess
    # In a real design, you'd have a more robust handshake.

    # Keep a CPU-based CIFAR-10 criterion for inference
    criterion = nn.CrossEntropyLoss()

    while steps_received < total_expected:
        # 1) Try to receive a loss from any GPU
        #    Because we’re rank=0, we can do a blocking receive with a "work or rank" approach.
        #    In advanced scenarios, we handle multiple GPU ranks.
        #    We'll do one rank at a time here for simplicity.
        #    If a GPU stops sending, we eventually break.
        try:
            loss_tensor = torch.zeros(1, dtype=torch.float32)
            dist.recv(loss_tensor, src=dist.any_source)  # blocks until some GPU sends
        except RuntimeError:
            print("[CPU 0] No more senders, stopping.")
            break

        # 2) Print the GPU loss
        gpu_loss = loss_tensor.item()
        print(f"[CPU 0] Received GPU loss: {gpu_loss:.4f}")

        # 3) Receive the shape and data for the last batch from that same GPU
        shape_tensor = torch.zeros(4, dtype=torch.long)
        dist.recv(shape_tensor, src=dist.any_source)
        shape_images = tuple(shape_tensor.tolist())  # e.g. (64, 3, 32, 32)

        flat_images = torch.zeros((int(torch.prod(shape_tensor)),), dtype=torch.float32)
        dist.recv(flat_images, src=dist.any_source)
        images_cpu = flat_images.view(shape_images)

        # 4) Receive labels
        shape_labels_t = torch.zeros(1, dtype=torch.long)
        dist.recv(shape_labels_t, src=dist.any_source)
        shape_labels = tuple(shape_labels_t.tolist())

        labels_cpu = torch.zeros(shape_labels, dtype=torch.long)
        dist.recv(labels_cpu, src=dist.any_source)

        # 5) CPU runs inference on the last batch with its local CPU model
        #    (which might be out-of-date unless we keep it in sync).
        #    Here we just do a fake forward with the CPU model as is.
        with torch.no_grad():
            out = cpu_model(images_cpu)
            _, preds = torch.max(out, dim=1)
            correct = (preds == labels_cpu).sum().item()
            batch_acc = float(correct) / float(labels_cpu.size(0))

        # 6) Send the accuracy back to the GPU
        acc_tensor = torch.tensor([batch_acc], dtype=torch.float32)
        dist.send(acc_tensor, dst=dist.any_source)  # send back to the GPU that asked

        steps_received += 1

    # ============ BONUS: Gather model from rank=1 and write to disk ============
    # We'll wait for rank=1 to send param shapes and data
    # In a real app, you'd do it each epoch or at the end of training.
    final_model = SimpleNet()
    final_model.eval()
    param_dict = final_model.state_dict()

    while True:
        # We'll read the shape. If we get shape=(-1), means GPU is done sending
        shape_t = torch.zeros(1, dtype=torch.long)
        dist.recv(shape_t, src=1)  # let's pick rank=1 as "primary"
        if shape_t.item() == -1:
            # done
            break

        shape_list = tuple(shape_t.tolist())
        flat_t = torch.zeros(int(torch.prod(shape_t)), dtype=torch.float32)
        dist.recv(flat_t, src=1)

        # We need a matching param name. This naive approach assumes we read them in order
        # matching final_model.named_parameters() iteration.
        # A robust approach might also send the name.
        for key in param_dict.keys():
            # if it's shape-compatible, fill it
            if list(param_dict[key].shape) == list(shape_list):
                param_dict[key].view(-1)[:] = flat_t
                break

    final_model.load_state_dict(param_dict)
    # Now we can write final_model to disk from CPU
    torch.save(final_model.state_dict(), f"{save_path}/final_model_cpu.pt")
    print("[CPU 0] Wrote final model to disk on CPU. Exiting.")

    dist.destroy_process_group()


###############################
# 4) Main: single entry point
###############################
def main():
    epochs = 10
    save_path = "./checkpoints"
    os.makedirs(save_path, exist_ok=True)

    # We have WORLD_SIZE total processes.
    # local_rank=0 -> run CPU logic
    # local_rank in [1..WORLD_SIZE-1] -> GPU training

    if LOCAL_RANK == 0:
        # CPU side
        run_cpu_process(world_size=WORLD_SIZE, epochs=epochs, save_path=save_path)

    else:
        # GPU side
        run_gpu_process(local_rank=LOCAL_RANK, world_size=WORLD_SIZE, epochs=epochs, save_path=save_path)


if __name__ == "__main__":
    main()
