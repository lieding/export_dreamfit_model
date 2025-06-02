from dataclasses import dataclass
import re
from layers_dreamfit import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor
from model_dreamfit import Flux, FluxParams
import torch
import importlib
import os
from safetensors import safe_open
from safetensors.torch import load_file as load_sft
from huggingface_hub import hf_hub_download
import torch.onnx
import onnxruntime
import numpy as np


def load_flow_model_by_type(name: str, device: str | torch.device = "cuda", hf_download: bool = True, lora_path=None, model_type='src.flux.model.Flux'):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path

    # with torch.device("meta" if ckpt_path is not None else device):
    with torch.device(device):
        
        model = Flux(configs[name].params).to(torch.bfloat16)

    if ckpt_path is not None:
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        
        if lora_path is not None:
            checkpoint = torch.load(lora_path, map_location=device)
            for k in checkpoint.keys():
                if "processor" not in k:
                    sd[k] = checkpoint[k]

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        # missing, unexpected = model.load_state_dict(sd, strict=False)
         
        print("Loading flow_model checkpoint")

    return model

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="XLabs-AI/flux-dev-fp8",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux-dev-fp8.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FP8"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]
        

def load_from_repo_id(repo_id, checkpoint_name):
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    sd = load_sft(ckpt_path, device='cpu')
    return sd
        
def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors
        
def load_checkpoint(local_path, repo_id, name):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    elif repo_id is not None and name is not None:
        print(f"Loading checkpoint {name} from repo id {repo_id}")
        checkpoint = load_from_repo_id(repo_id, name)
    else:
        raise ValueError(
            "LOADING ERROR: you must specify local_path or repo_id with name in HF to download"
        )
    return checkpoint


def update_model_with_lora(model: Flux, checkpoint, lora_weight, network_alpha, double_blocks, single_blocks):
    rank =  get_lora_rank(checkpoint)

    print("rank ", rank)
    lora_attn_procs = {}
        
    if double_blocks is None:
        double_blocks_idx = list(range(19))
    else:
        double_blocks_idx = [int(idx) for idx in double_blocks.split(",")]

    if single_blocks is None:
        single_blocks_idx = list(range(38))
    elif single_blocks is not None:
        if single_blocks == "":
            single_blocks_idx = []
        else:
            single_blocks_idx = [int(idx) for idx in single_blocks.split(",")]

    # load lora ckpt for modulation
    dit_state_dict = model.state_dict()
    modulation_lora_state_dict = {}
    for name in dit_state_dict.keys():
        if 'lin_lora' in name:
            modulation_lora_state_dict[name] = checkpoint[name]
    missing, unexpected = model.load_state_dict(modulation_lora_state_dict, strict=False)
    print('missing parameters:', missing)
    print('unexpected parameters:', unexpected)

    # load lora ckpt for attn processor
    for name, attn_processor in model.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks") and layer_index in double_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                dim=3072, rank=rank, network_alpha=network_alpha
            )

            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

            lora_attn_procs[name].load_state_dict(lora_state_dict)
            lora_attn_procs[name].to("cpu", dtype=torch.bfloat16)

        elif name.startswith("single_blocks") and layer_index in single_blocks_idx:
            print("setting LoRA Processor for", name)
            lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                dim=3072, rank=rank, network_alpha=network_alpha
            )

            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

            lora_attn_procs[name].load_state_dict(lora_state_dict)
            lora_attn_procs[name].to("cpu", dtype=torch.bfloat16)

        else:
            lora_attn_procs[name] = attn_processor
            
    model.set_attn_processor(lora_attn_procs)

def set_lora(model, local_path: str = None, repo_id: str = None, name: str = None, lora_weight: float = 0.7, network_alpha=None, double_blocks=None, single_blocks=None): # type: ignore
    checkpoint = load_checkpoint(local_path, repo_id, name)

    update_model_with_lora(model, checkpoint, lora_weight, network_alpha, double_blocks, single_blocks)

def Load_model(lora_path: str="https://huggingface.co/bytedance-research/Dreamfit/resolve/main/flux_i2i.bin?download=true", model_path: str="https://huggingface.co/Shakker-Labs/AWPortrait-FL/resolve/main/AWPortrait-FL-fp8.safetensors?download=true"):
    model = load_flow_model_by_type("flux-dev", device="cpu", lora_path=lora_path, model_type=model_path)
    set_lora(model, lora_path, "", "", 1.0, 16, None, None)
    print("Exporting model to ONNX...")
    export_model_to_onnx(model, "flux_model.onnx")
    print("Model exported to flux_model.onnx")
    return model


def export_model_to_onnx(model, onnx_filepath):
    """Exports the given model to ONNX format.

    Args:
        model: The PyTorch model to export.
        onnx_filepath: The path to save the ONNX model.
    """
    params = model.params

    # Create dummy inputs
    img_seq_len = params.axes_dim[1] * params.axes_dim[2]
    img = torch.randn(1, img_seq_len, params.in_channels, dtype=torch.bfloat16)
    img_ids = torch.zeros(1, img_seq_len, params.hidden_size // params.num_heads, dtype=torch.bfloat16)

    txt_seq_len = 77  # Standard CLIP sequence length
    txt = torch.randn(1, txt_seq_len, params.context_in_dim, dtype=torch.bfloat16)
    txt_ids = torch.zeros(1, txt_seq_len, params.hidden_size // params.num_heads, dtype=torch.bfloat16)

    timesteps = torch.randn(1, dtype=torch.bfloat16)
    y = torch.randn(1, params.vec_in_dim, dtype=torch.bfloat16)

    guidance = None
    if params.guidance_embed:
        guidance = torch.randn(1, dtype=torch.bfloat16)

    dummy_inputs = (img, img_ids, txt, txt_ids, timesteps, y)
    input_names = ["img", "img_ids", "txt", "txt_ids", "timesteps", "y"]

    if guidance is not None:
        dummy_inputs += (guidance,)
        input_names.append("guidance")

    output_names = ["output"]  # Replace with actual output names if known

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_filepath,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        # example_outputs=model(*dummy_inputs) # Optional: for output shape inference
    )
    print(f"Model exported to {onnx_filepath}")


def verify_onnx_model(onnx_filepath, params):
    """Verifies the ONNX model by running inference with dummy inputs.

    Args:
        onnx_filepath: Path to the ONNX model file.
        params: Model parameters from the original PyTorch model.
    """
    try:
        print("Verifying ONNX model...")

        # Create dummy inputs as NumPy arrays (float32)
        img_seq_len = params.axes_dim[1] * params.axes_dim[2]
        img = np.random.randn(1, img_seq_len, params.in_channels).astype(np.float32)
        img_ids = np.zeros((1, img_seq_len, params.hidden_size // params.num_heads), dtype=np.float32)

        txt_seq_len = 77  # Standard CLIP sequence length
        txt = np.random.randn(1, txt_seq_len, params.context_in_dim).astype(np.float32)
        txt_ids = np.zeros((1, txt_seq_len, params.hidden_size // params.num_heads), dtype=np.float32)

        timesteps = np.random.randn(1).astype(np.float32)
        y = np.random.randn(1, params.vec_in_dim).astype(np.float32)

        ort_inputs = {
            "img": img,
            "img_ids": img_ids,
            "txt": txt,
            "txt_ids": txt_ids,
            "timesteps": timesteps,
            "y": y,
        }

        if params.guidance_embed:
            guidance = np.random.randn(1).astype(np.float32)
            ort_inputs["guidance"] = guidance

        # Create ONNX runtime session
        ort_session = onnxruntime.InferenceSession(onnx_filepath)
        print(f"ONNX session created for {onnx_filepath}")

        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)

        print("ONNX model verification successful.")
        print(f"Output shape: {ort_outputs[0].shape}")

    except Exception as e:
        print(f"Error during ONNX model verification: {e}")
        print("Please check for potential dtype mismatches (e.g., bfloat16 vs float32) or model export issues.")

if __name__ == "__main__":
    model = Load_model()
    if model: # If model loaded successfully
        verify_onnx_model("flux_model.onnx", model.params)
