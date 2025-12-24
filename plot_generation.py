import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator


def smooth(scalars, weight=0.8):
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def format_k(x, pos):
    if x >= 1000:
        return f"{x*1e-3:g}k"
    return f"{x:g}"


def main():
    root_log_dir_path = "./runs"
    output_folder_path = "./graphs"
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    all_train_data = []
    all_val_data = []
    combined_data = {}  # To store both for the final graph

    # --- PART 1: Process Folders & Individual Graphs ---
    for subdir in sorted(os.listdir(root_log_dir_path)):
        log_dir = os.path.join(root_log_dir_path, subdir)
        if not os.path.isdir(log_dir):
            continue

        individual_plot_path = os.path.join(output_folder_path, f"plot_{subdir}.png")
        label_name = subdir[11:] if len(subdir) > 11 else subdir

        print(f"Processing: {subdir}...")

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        tags = ea.Tags()["scalars"]

        train_acc_tag = next(
            (t for t in tags if "train" in t.lower() and "acc" in t.lower()), None
        )
        val_acc_tag = next(
            (t for t in tags if "val" in t.lower() and "acc" in t.lower()), None
        )

        # Temporary storage for this specific model
        model_entry = {}

        if train_acc_tag:
            t_steps, t_values = zip(
                *[(s.step, s.value) for s in ea.Scalars(train_acc_tag)]
            )
            t_smooth = smooth(t_values)
            all_train_data.append((label_name, t_steps, t_smooth))
            model_entry["train"] = (t_steps, t_smooth)

        if val_acc_tag:
            v_steps, v_values = zip(
                *[(s.step, s.value) for s in ea.Scalars(val_acc_tag)]
            )
            v_smooth = smooth(v_values)
            all_val_data.append((label_name, v_steps, v_smooth))
            model_entry["val"] = (v_steps, v_smooth)

        # Populate the combined dictionary for the final plot
        if model_entry:
            combined_data[label_name] = model_entry

        # ONLY create the individual plot if it doesn't exist
        if os.path.exists(individual_plot_path):
            print(f"  [>] Individual plot already exists. Skipping image generation.")
        else:
            if not train_acc_tag and not val_acc_tag:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            if train_acc_tag:
                plt.plot(
                    t_steps, t_values, alpha=0.3, color="#1f77b4", label="Train (Raw)"
                )
                plt.plot(
                    t_steps,
                    t_smooth,
                    color="#1f77b4",
                    linewidth=2,
                    label="Train (Smooth)",
                )

            if val_acc_tag:
                plt.plot(
                    v_steps, v_values, alpha=0.3, color="#ff7f0e", label="Val (Raw)"
                )
                plt.plot(
                    v_steps,
                    v_smooth,
                    color="#ff7f0e",
                    linewidth=2,
                    label="Val (Smooth)",
                )

            ax.xaxis.set_major_formatter(FuncFormatter(format_k))
            ax.set_title(f"Accuracy: {label_name.title()}", fontsize=14)
            ax.set_xlabel("Steps", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(individual_plot_path)
            plt.close()
            print(f"  [+] Created individual plot.")

    # --- PART 2: Global Comparison Graph - TRAINING ---
    train_comp_path = os.path.join(output_folder_path, "TOTAL_TRAIN_ACCURACY.png")
    if not os.path.exists(train_comp_path) and all_train_data:
        plt.figure(figsize=(12, 7))
        for name, steps, values in all_train_data:
            plt.plot(steps, values, label=name, linewidth=1.5, alpha=0.8)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_k))
        plt.title("All Models: Training Accuracy Comparison", fontsize=16)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(train_comp_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"\nSaved Training Comparison: {train_comp_path}")

    # --- PART 3: Global Comparison Graph - VALIDATION ---
    val_comp_path = os.path.join(output_folder_path, "TOTAL_VAL_ACCURACY.png")
    if not os.path.exists(val_comp_path) and all_val_data:
        plt.figure(figsize=(12, 7))
        for name, steps, values in all_val_data:
            plt.plot(steps, values, label=name, linewidth=2)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_k))
        plt.title("All Models: Validation Accuracy Comparison", fontsize=16)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(val_comp_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved Validation Comparison: {val_comp_path}")

    # --- PART 4: Global Comparison Graph - COMBINED ---
    comb_total_path = os.path.join(output_folder_path, "TOTAL_TRAIN_VAL_COMPARISON.png")
    if not os.path.exists(comb_total_path) and combined_data:
        plt.figure(figsize=(14, 8))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (name, data) in enumerate(combined_data.items()):
            color = colors[i % len(colors)]
            if "train" in data:
                plt.plot(
                    data["train"][0],
                    data["train"][1],
                    label=f"{name} (Train)",
                    color=color,
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.2,
                )
            if "val" in data:
                plt.plot(
                    data["val"][0],
                    data["val"][1],
                    label=f"{name} (Val)",
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                )

        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_k))
        plt.title("All Models: Training (Solid) vs Validation (Dashed)", fontsize=16)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(comb_total_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved Total Train/Val Comparison: {comb_total_path}")


if __name__ == "__main__":
    main()
