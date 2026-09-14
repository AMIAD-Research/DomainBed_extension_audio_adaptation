import time 
import csv
from datetime import datetime
from collections import defaultdict
import os
import torch


def save_elapsed_time(csv_path, job_info, start_time, step): 
    elapsed_sec = time.monotonic() - start_time
    with open(csv_path, mode='a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            f"step {step}", 
            f"global time {round(elapsed_sec, 3)}s",
            job_info
        ])
        if (step>0):
            writer.writerow([])


def save_accuracy_per_class(path, results): 
    # Separate global info and env results
    global_keys = ['step', 'epoch', 'SNR']
    global_info = {k: results[k] for k in global_keys if k in results}

    # Group env results
    grouped = {"in": defaultdict(list), "out": defaultdict(list), "uda": defaultdict(list)}

    for key, value in results.items():
        if key in global_keys:
            continue  # skip global info
        env_part, class_part = key.split("/")          # "env0_in", "acc_class_0"
        env_name, direction = env_part.split("_", 1)  # "env0", "in/out"
        class_name = class_part.replace("acc_", "")   # "class_0"
        grouped[direction][class_name].append(f"{env_name}: {value}")

    with open(path, "a") as f:
        # 2a: Write global info first
        for k, v in global_info.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        # 2b: Write grouped env results
        for direction in ["in", "out"]:
            f.write(f"=== {direction.upper()} ===\n")
            for class_name, entries in grouped[direction].items():
                f.write(f"{class_name}: " + ", ".join(entries) + "\n")
            f.write("\n")



from collections import Counter
import pandas as pd

def build_table(dataset_list):

    # print("building tables...")
    rows = []

    for domain_id, domain in enumerate(dataset_list):
        # print(domain_id)
        dataset = domain[0]
        n = len(dataset)

        counter = Counter()

        for i in range(n):
            # print(f'in iteration {i}')
            _, label = dataset[i]
            # print(label)
            counter[label] += 1
            # print(counter)

        row = {"Domain": domain_id}

        for cls in sorted(counter):
            count = counter[cls]
            prop = count / n
            row[f"Class {cls}"] = f"{count} ({prop:.3f})"

        rows.append(row)

    return pd.DataFrame(rows).set_index("Domain")


def print_class_stats(dataset_list, name):
    print(f"\n===== {name} =====")
    table = build_table(dataset_list)
    print(table)



def write_class_stats(file, dataset_list, name):
    table = build_table(dataset_list)

    file.write(f"\n===== {name} =====\n\n")
    file.write(table.to_string())
    file.write("\n")


def statistics_splits(out_dir, in_, out, uda_all, uda=None):

    print_class_stats(in_, "IN")
    print_class_stats(out, "OUT")
    print_class_stats(uda_all, "UDA_ALL")

    if uda is not None:
        print_class_stats(uda, "UDA")

    os.makedirs(out_dir, exist_ok=True)


    filepath = os.path.join(out_dir, "split_statistics.txt")

    with open(filepath, "w") as f:
        write_class_stats(f, in_, "IN")
        write_class_stats(f, out, "OUT")
        write_class_stats(f, uda_all, "UDA_ALL")

        if uda is not None:
            write_class_stats(f, uda, "UDA")



@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()

    embeddings = []
    labels = []

    for batch in loader:
        x, y = batch[:2]

        x = x.to(device)

        z = model(x)

        embeddings.append(z.cpu())
        labels.append(y.cpu())

    return (
        torch.cat(embeddings, dim=0),
        torch.cat(labels, dim=0),
    )



def compute_jacobian_norm(model, x):
    """
    Frobenius norm of d logits / d input for each sample.

    Returns:
        jacobian_norm: [B]
    """
    x = x.detach().clone().requires_grad_(True)

    logits = model.predict(x)
    B, C = logits.shape

    jacobian_norm_sq = torch.zeros(
        B,
        device=x.device,
        dtype=x.dtype
    )

    for c in range(C):
        grad = torch.autograd.grad(
            outputs=logits[:, c].sum(),
            inputs=x,
            create_graph=False,
            retain_graph=(c < C - 1),
        )[0]

        # || d f_c / dx ||_2^2 for each sample
        jacobian_norm_sq += grad.flatten(start_dim=1).pow(2).sum(dim=1)

    return torch.sqrt(jacobian_norm_sq)