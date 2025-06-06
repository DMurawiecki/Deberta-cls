import os

import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer

ckpt_path = "../models/model/epoch=02-val_loss=1.0025.ckpt"
checkpoint_folder = "../models/model"
output_dir = "../models/model/hf_pretrained"
os.makedirs(output_dir, exist_ok=True)


def main():
    print(f"Loading Lightning checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" not in ckpt:
        raise RuntimeError(f"No 'state_dict' found in checkpoint: {ckpt_path}")
    lightning_state = ckpt["state_dict"]

    print(f"Instantiating HF model from: {checkpoint_folder}")
    hf_model = AutoModelForMultipleChoice.from_pretrained(
        checkpoint_folder, local_files_only=True
    )

    print("Preparing state_dict for HF model...")
    new_state_dict = {}
    prefix = "model."
    len_prefix = len(prefix)
    for key, value in lightning_state.items():
        if key.startswith(prefix):
            new_key = key[len_prefix:]
            new_state_dict[new_key] = value

    print("Loading weights into HF model...")
    missing, unexpected = hf_model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print("WARNING: Missing keys in HF model:")
        for k in missing:
            print("  -", k)
    if unexpected:
        print("WARNING: Unexpected keys from Lightning checkpoint:")
        for k in unexpected:
            print("  -", k)

    print(f"Saving HuggingFace model to: {output_dir}")
    hf_model.save_pretrained(output_dir)

    print("Saving tokenizer *into the same folder*...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder, local_files_only=True)
    tokenizer.save_pretrained(output_dir)

    print("Done. HF folder now contains:")
    for fname in sorted(os.listdir(output_dir)):
        print("  •", fname)
    print()
    print("You can now run ONNX export against this folder:")
    print(f"  bash convert_to_onnx.sh {output_dir} ../model_onnx/")


if __name__ == "__main__":
    main()
