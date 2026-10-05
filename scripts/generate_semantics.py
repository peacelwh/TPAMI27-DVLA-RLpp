"""Generate intrinsic / nuisance semantic banks from labeled support images.

A vision-language generator (Qwen2.5-VL by default) receives the labeled
supports of one class and returns concise *affirmative* phrases in two roles:

    intrinsic : visible object properties that define the class
    nuisance  : visible scene context, co-occurring objects, occlusions or
                acquisition cues (may be empty)

Each role has local phrases (at most ``--max_local``) and one global summary.
Phrases are encoded with the frozen CLIP ViT-B/16 text encoder and written to
``data/semantic/<dataset>_banks.pth`` in the format documented in
``src/dvla_rlpp/data/semantic.py``. Raw phrases are also dumped as JSON for
inspection. Query images never enter this script.

Example:
    python scripts/generate_semantics.py --dataset miniImageNet --splits base,val,novel --gpu 0
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--data_root", type=str, default=os.path.join(ROOT, "data", "datasets"))
parser.add_argument("--splits", type=str, default="base,val,novel")
parser.add_argument("--generator", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
parser.add_argument("--text_encoder", type=str, default="ViT-B/16")
parser.add_argument("--prompt", type=str, default=os.path.join(ROOT, "data", "prompts", "intrinsic_nuisance_prompt.txt"))
parser.add_argument("--images_per_class", type=int, default=5, help="labeled supports shown to the generator")
parser.add_argument("--per_support", action="store_true", help="additionally generate one entry per support image")
parser.add_argument("--max_local", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="")
parser.add_argument("--dump_json", type=str, default="")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dvla_rlpp.data.datasets import class_display_name, dataset_root  # noqa: E402


def load_generator(name):
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as e:
        raise SystemExit("pip install 'transformers>=4.49' accelerate qwen-vl-utils   (needed for generation)") from e
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(name)
    return model, processor


def load_text_encoder(name, device):
    try:
        import clip
    except ImportError as e:
        raise SystemExit("pip install git+https://github.com/openai/CLIP.git   (needed for text encoding)") from e
    model, _ = clip.load(name, device=device)
    model.eval()

    @torch.no_grad()
    def encode(phrases):
        if not phrases:
            return torch.zeros(0, 512)
        tokens = clip.tokenize(phrases, truncate=True).to(device)
        feats = model.encode_text(tokens).float()
        return torch.nn.functional.normalize(feats, dim=-1).cpu()
    return encode


def generate(model, processor, images, class_name, prompt, max_new_tokens):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise SystemExit("pip install qwen-vl-utils") from e
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt.replace("{class_name}", class_name.replace("_", " "))})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def parse_response(text, max_local):
    """Extract {"intrinsic": {...}, "nuisance": {...}} from the generator output."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    data = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = {}

    def clean(lst):
        seen, out = set(), []
        for p in lst or []:
            p = re.sub(r"\s+", " ", str(p)).strip().strip(".").lower()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out[:max_local]

    def role(name):
        r = data.get(name, {}) if isinstance(data, dict) else {}
        local = clean(r.get("local", [])) if isinstance(r, dict) else []
        glob = r.get("global", "") if isinstance(r, dict) else ""
        glob = re.sub(r"\s+", " ", str(glob)).strip()
        return {"local": local, "global": glob}
    return {"intrinsic": role("intrinsic"), "nuisance": role("nuisance")}


def encode_entry(entry, encode):
    return {
        "intrinsic": {"local": encode(entry["intrinsic"]["local"]),
                      "global": encode([entry["intrinsic"]["global"]] if entry["intrinsic"]["global"] else [])},
        "nuisance": {"local": encode(entry["nuisance"]["local"]),
                     "global": encode([entry["nuisance"]["global"]] if entry["nuisance"]["global"] else [])},
    }


def main():
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = open(args.prompt).read()
    out_path = args.out or os.path.join(ROOT, "data", "semantic", f"{args.dataset}_banks.pth")
    dump_path = args.dump_json or os.path.join(ROOT, "results", "semantic", f"{args.dataset}_phrases.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)

    gen_model, processor = load_generator(args.generator)
    encode = load_text_encoder(args.text_encoder, device)

    classes, supports, raw = {}, {}, {}
    for split in args.splits.split(","):
        split_dir = os.path.join(dataset_root(args.data_root, args.dataset), split)
        for folder in sorted(os.listdir(split_dir)):
            cdir = os.path.join(split_dir, folder)
            if not os.path.isdir(cdir):
                continue
            name = class_display_name(args.dataset, folder)
            files = sorted(f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
            chosen = random.sample(files, min(args.images_per_class, len(files)))
            images = [Image.open(os.path.join(cdir, f)).convert("RGB") for f in chosen]
            text = generate(gen_model, processor, images, name, prompt, args.max_new_tokens)
            entry = parse_response(text, args.max_local)
            raw[name] = {"supports": [os.path.join(split, folder, f) for f in chosen], "phrases": entry, "raw": text}
            enc = encode_entry(entry, encode)
            enc["name"] = encode([f"a photo of a {name.replace('_', ' ')}"])[0]
            classes[name] = enc
            print(f"[{split}] {name}: {len(entry['intrinsic']['local'])} intrinsic / {len(entry['nuisance']['local'])} nuisance phrases")
            if args.per_support:
                for f in chosen:
                    rel = os.path.join(folder, f)
                    img = Image.open(os.path.join(cdir, f)).convert("RGB")
                    e = parse_response(generate(gen_model, processor, [img], name, prompt, args.max_new_tokens), args.max_local)
                    supports[rel] = encode_entry(e, encode)
                    raw.setdefault("_supports", {})[rel] = e

    meta = {"dataset": args.dataset, "generator": args.generator, "text_encoder": args.text_encoder,
            "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(), "seed": args.seed,
            "images_per_class": args.images_per_class, "max_local": args.max_local}
    torch.save({"text_dim": 512, "classes": classes, "supports": supports, "meta": meta}, out_path)
    with open(dump_path, "w") as f:
        json.dump(raw, f, indent=1, ensure_ascii=False)
    print(f"saved {len(classes)} class banks and {len(supports)} support entries to {out_path}")


if __name__ == "__main__":
    main()
