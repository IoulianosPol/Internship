import os
import sys
import argparse
import datetime
import gc
import torch
import torch.nn.functional as F
import transformers
import wandb
import huggingface_hub
import networkx as nx
import matplotlib.pyplot as plt
from acdc.acdc_graphics import show

from functools import partial

# --- Bypasses & Fixes ---
hf_cache_dir = os.environ.get("HF_HOME", os.path.join(os.getcwd(), "hf_cache"))
os.makedirs(hf_cache_dir, exist_ok=True)
transformers.TRANSFORMERS_CACHE = hf_cache_dir
os.environ["TRANSFORMERS_USE_FAST"] = "True"

import transformer_lens

sys.modules['transformer_lens.HookedTransformerConfig'] = transformer_lens
sys.modules['transformer_lens.HookedTransformer'] = transformer_lens
sys.modules['transformer_lens.ActivationCache'] = transformer_lens
sys.modules['transformer_lens.FactoredMatrix'] = transformer_lens

import types

dummy_typeguard = types.ModuleType("typeguard")
dummy_typeguard.typechecked = lambda func: func
sys.modules["typeguard"] = dummy_typeguard


class DummyTensorType:
    def __getitem__(self, key):
        return self


dummy_torchtyping = types.ModuleType("torchtyping")
dummy_torchtyping.TensorType = DummyTensorType()
dummy_torchtyping.patch_typeguard = lambda *args, **kwargs: None
sys.modules["torchtyping"] = dummy_torchtyping

from transformer_lens.HookedTransformer import HookedTransformer

try:
    from acdc.tracr_task.utils import get_all_tracr_things
except Exception as e:
    print(f"Could not import `tracr` because {e}; tracr tasks will be unavailable.")

from acdc.docstring.utils import get_all_docstring_things
from acdc.acdc_utils import reset_network, shuffle_tensor, ct, MatchNLLMetric, kl_divergence, negative_log_probs
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.ioi.utils import get_all_ioi_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.acdc_graphics import show
from acdc.docstring.utils import AllDataThings
import wandb
import IPython
from IPython.display import Image, display
import torch
import gc
from tqdm import tqdm
import networkx as nx
import huggingface_hub
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import yaml
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)

try:
    from acdc.tracr_task.utils import (
        get_all_tracr_things,
        get_tracr_model_input_and_tl_model,
    )
except Exception as e:
    print(f"Could not import `tracr` because {e}; the rest of the file should work but you cannot use the tracr tasks")

from acdc.docstring.utils import get_all_docstring_things
from acdc.acdc_utils import (
    make_nd_dict,
    reset_network,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment

from acdc.acdc_utils import (
    kl_divergence,
)
from acdc.ioi.utils import (
    get_all_ioi_things,
    get_gpt2_small,
)
from acdc.induction.utils import (
    get_all_induction_things,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
    get_model
)
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.acdc_graphics import (
    build_colorscheme,
    show,
)

torch.autograd.set_grad_enabled(False)
print("Environment setup successfully.")

# --- Tokenizer Patch ---
orig_auto = transformers.AutoTokenizer.from_pretrained
orig_gpt2 = transformers.GPT2Tokenizer.from_pretrained
orig_gpt2_fast = transformers.GPT2TokenizerFast.from_pretrained


def universal_tokenizer_patch(orig_fn):
    def patched_fn(pretrained_model_name_or_path, *args, **kwargs):
        if isinstance(pretrained_model_name_or_path, str) and "redwood" in pretrained_model_name_or_path:
            pretrained_model_name_or_path = "gpt2"
        kwargs['use_fast'] = True
        return orig_fn(pretrained_model_name_or_path, *args, **kwargs)

    return patched_fn


transformers.AutoTokenizer.from_pretrained = universal_tokenizer_patch(orig_auto)
transformers.GPT2Tokenizer.from_pretrained = universal_tokenizer_patch(orig_gpt2)
transformers.GPT2TokenizerFast.from_pretrained = universal_tokenizer_patch(orig_gpt2_fast)


# --- Functions ---
def get_model(device):
    tl_model = HookedTransformer.from_pretrained(
        "gpt2",
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        dtype=torch.float16,
        device=device,
    )
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    tl_model.set_use_hook_mlp_in(True)
    return tl_model


def get_validation_data(num_examples=None, seq_len=None, device=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname, map_location=device).long()
    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]


def get_mask_repeat_candidates(num_examples=None, seq_len=None, device=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(mask_repeat_candidates_fname, map_location=device)
    mask_repeat_candidates.requires_grad = False
    if num_examples is None:
        return mask_repeat_candidates
    else:
        return mask_repeat_candidates[:num_examples, :seq_len]


def get_all_induction_things(num_examples, seq_len, device, data_seed=42, metric="kl_div",
                             return_one_element=True) -> AllDataThings:
    tl_model = get_model(device=device)
    validation_data_orig = get_validation_data(device=device)
    vocab_size = tl_model.cfg.d_vocab
    safe_token = tl_model.tokenizer.eos_token_id
    validation_data_orig[validation_data_orig >= vocab_size] = safe_token

    mask_orig = get_mask_repeat_candidates(num_examples=None, device=device)
    validation_slice = slice(0, num_examples)
    validation_data = validation_data_orig[validation_slice, :seq_len].contiguous()
    validation_labels = validation_data_orig[validation_slice, 1:seq_len + 1].contiguous()
    validation_mask = mask_orig[validation_slice, :seq_len].contiguous()
    validation_patch_data = shuffle_tensor(validation_data, seed=data_seed).contiguous()

    test_slice = slice(num_examples, num_examples * 2)
    test_data = validation_data_orig[test_slice, :seq_len].contiguous()
    test_labels = validation_data_orig[test_slice, 1:seq_len + 1].contiguous()
    test_mask = mask_orig[test_slice, :seq_len].contiguous()
    test_patch_data = shuffle_tensor(test_data, seed=data_seed).contiguous()

    with torch.no_grad():
        base_val_logprobs = F.log_softmax(tl_model(validation_data), dim=-1).detach()
        base_test_logprobs = F.log_softmax(tl_model(test_data), dim=-1).detach()

    if metric == "kl_div":
        validation_metric = partial(kl_divergence, base_model_logprobs=base_val_logprobs,
                                    mask_repeat_candidates=validation_mask, last_seq_element_only=False,
                                    return_one_element=return_one_element)
    elif metric == "nll":
        validation_metric = partial(negative_log_probs, labels=validation_labels,
                                    mask_repeat_candidates=validation_mask, last_seq_element_only=False)
    elif metric == "match_nll":
        validation_metric = MatchNLLMetric(labels=validation_labels, base_model_logprobs=base_val_logprobs,
                                           mask_repeat_candidates=validation_mask, last_seq_element_only=False)
    else:
        raise ValueError(f"Unknown metric {metric}")

    test_metrics = {
        "kl_div": partial(kl_divergence, base_model_logprobs=base_test_logprobs, mask_repeat_candidates=test_mask,
                          last_seq_element_only=False)}
    return AllDataThings(tl_model=tl_model, validation_metric=validation_metric, validation_data=validation_data,
                         validation_labels=validation_labels, validation_mask=validation_mask,
                         validation_patch_data=validation_patch_data, test_metrics=test_metrics, test_data=test_data,
                         test_labels=test_labels, test_mask=test_mask, test_patch_data=test_patch_data)


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Launch ACDC runs.")
    task_choices = ['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan']
    parser.add_argument('--task', type=str, default='induction', choices=task_choices)
    parser.add_argument('--threshold', type=float, default=0.5623)
    parser.add_argument('--first-cache-cpu', type=str, default="True")
    parser.add_argument('--second-cache-cpu', type=str, default="True")
    parser.add_argument('--zero-ablation', action='store_true')
    parser.add_argument('--using-wandb', action='store_true')
    parser.add_argument('--wandb-entity-name', type=str, default="remix_school-of-rock")
    parser.add_argument('--wandb-group-name', type=str, default="default")
    parser.add_argument('--wandb-project-name', type=str, default="acdc")
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default="./wandb_cache")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument('--indices-mode', type=str, default="normal")
    parser.add_argument('--names-mode', type=str, default="normal")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--reset-network', type=int, default=0)
    parser.add_argument('--metric', type=str, default="kl_div")
    parser.add_argument('--torch-num-threads', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument("--max-num-epochs", type=int, default=100_000)
    parser.add_argument('--single-step', action='store_true')
    parser.add_argument("--abs-value-threshold", action='store_true')

    # HARDCODED ARGS OVRERRIDE (Όπως στο notebook)
    args = parser.parse_args(
        [line.strip() for line in r"""--task=induction\
--threshold=0.8\
--indices-mode=reverse\
--first-cache-cpu=True\
--second-cache-cpu=True\
--max-num-epochs=100000""".split("\\\n")]
    )

    return args


def main():
    args = parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    torch.manual_seed(args.seed)

    TASK = args.task
    ONLINE_CACHE_CPU = args.first_cache_cpu.lower() == "true"
    CORRUPTED_CACHE_CPU = args.second_cache_cpu.lower() == "true"

    if not os.path.exists("ims/"):
        os.makedirs("ims/")

    print(f"Task: {TASK} with threshold: {args.threshold}...")
    second_metric = None
    use_pos_embed = TASK.startswith("tracr")

    if TASK == "ioi":
        things = get_all_ioi_things(num_examples=40, device=args.device, metric_name=args.metric)
    elif TASK == "induction":
        things = get_all_induction_things(num_examples=2, seq_len=300, device=args.device, metric=args.metric)
    elif TASK == "docstring":
        things = get_all_docstring_things(num_examples=50, seq_len=41, device=args.device, metric_name=args.metric,
                                          correct_incorrect_wandb=True)
    elif TASK == "greaterthan":
        things = get_all_greaterthan_things(num_examples=100, metric_name=args.metric, device=args.device)
    else:
        raise ValueError(f"Task {TASK} specific setup not fully integrated in script.")

    print("\nData loaded successfully!")

    validation_metric = things.validation_metric
    toks_int_values = things.validation_data
    toks_int_values_other = things.validation_patch_data
    tl_model = things.tl_model

    if args.reset_network:
        reset_network(TASK, args.device, tl_model)

    tl_model.reset_hooks()
    gc.collect()
    torch.cuda.empty_cache()

    WANDB_RUN_NAME = args.wandb_run_name if args.wandb_run_name else f"{ct()}_{args.threshold}"

    exp = TLACDCExperiment(
        model=tl_model,
        threshold=args.threshold,
        using_wandb=args.using_wandb,
        wandb_entity_name=args.wandb_entity_name,
        wandb_project_name=args.wandb_project_name,
        wandb_run_name=WANDB_RUN_NAME,
        wandb_group_name=args.wandb_group_name,
        wandb_notes="HPC Run",
        wandb_dir=args.wandb_dir,
        wandb_mode=args.wandb_mode,
        wandb_config=args,
        zero_ablation=args.zero_ablation,
        abs_value_threshold=args.abs_value_threshold,
        ds=toks_int_values,
        ref_ds=toks_int_values_other,
        metric=validation_metric,
        second_metric=second_metric,
        verbose=True,
        indices_mode=args.indices_mode,
        names_mode=args.names_mode,
        corrupted_cache_cpu=CORRUPTED_CACHE_CPU,
        hook_verbose=False,
        online_cache_cpu=ONLINE_CACHE_CPU,
        add_sender_hooks=True,
        use_pos_embed=use_pos_embed,
        add_receiver_hooks=False,
        remove_redundant=False,
        show_full_index=use_pos_embed,
    )

    # Instead of reloading the data, we extract it directly
    # from the 'things' object prepared by ACDC.
    # val_data contains the texts in numerical format (token IDs).
    val_data = things.validation_data
    # mask is a boolean array (True/False).
    # True means: "This token is the [B] in a pattern [A][B]...[A][B]".
    mask = things.validation_mask

    # Loop through all 10 (or the specified number of) prompts
    for prompt_idx in range(len(val_data)):
        # Get the list of token IDs for the specific prompt
        prompt_tokens = val_data[prompt_idx]
        # Convert the numbers (token IDs) back into readable text
        prompt_text = tl_model.to_string(prompt_tokens)

        print("\n" + "=" * 60)
        print(f"--- Real Text (Prompt {prompt_idx}) ---")
        print(prompt_text)

        print(f"\n--- Induction Targets (Mask) for Prompt {prompt_idx} ---")

        # Counter to track how many induction tokens were found
        induction_count = 0
        # zip() pairs each token_id with its corresponding True/False value from the mask.
        # enumerate() gives us the position (index 'i') of this token within the text.
        for i, (token_id, is_induction_target) in enumerate(zip(prompt_tokens, mask[prompt_idx])):
            if is_induction_target:
                # If the mask at this position is True (meaning we expect the model to copy)
                print(f"Position {i}: The model was expected to predict token '{tl_model.to_string(token_id)}'")
                induction_count += 1

        if induction_count == 0:
            # If the text had no repeating patterns at all
            print("No repeating induction patterns were found in this prompt.")

    print("Model:")
    print(tl_model.cfg)
    print("Initial Edges:", exp.count_no_edges())
    all_nodes = [
        node
        for receiver_dict in exp.corr.graph.values()
        for node in receiver_dict.values()
    ]

    print("Initial Nodes:", len(all_nodes))
    print("-" * 50)
    print(exp.corr.nodes())
    print("Initial connections")
    for edge_tuple, edge in exp.corr.all_edges().items():
        if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
            receiver_name, receiver_idx, sender_name, sender_idx = edge_tuple

            rec_str = f"{receiver_name} {receiver_idx.hashable_tuple}"
            send_str = f"{sender_name} {sender_idx.hashable_tuple}"

            print(f"From: {send_str}  --->  To: {rec_str}")
    show(exp.corr, "ims/initial_full_network.png", show_full_index=False)
    print("Saved initial network image to ims/initial_full_network.png")

    print("Initial Graph")

    corr = TLACDCCorrespondence.setup_from_model(tl_model, use_pos_embed=False)

    for (recv_name, recv_idx, send_name, send_idx), edge in corr.all_edges().items():
        if edge.present:
            edge.effect_size = 1.0

    fname = "ims/official_acdc_114.png"
    show(
        correspondence=corr,
        fname=fname,
        show_full_index=False,
        remove_qkv=False,
        show_placeholders=False,

    )

    display(Image(fname))

    exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    last_edge_count = exp.count_no_edges()

    for i in range(args.max_num_epochs):
        exp.step(testing=False)
        current_edge_count = exp.count_no_edges()

        if current_edge_count < last_edge_count:
            #print(f"Edge removed! New set: {current_edge_count}")
            #fname = f"ims/img_pruned_{i + 1}.png"
            #show(exp.corr, fname=fname, show_full_index=False)
            last_edge_count = current_edge_count

        print(f"Epoch {i} | Edges remaining: {current_edge_count}")

        if i == 0:
            exp.save_edges("edges.pkl")

        if exp.current_node is None or args.single_step:
            show(exp.corr, f"ims/ACDC_img_{exp_time}.png")
            print(f"Finished. Final graph saved to ims/ACDC_img_{exp_time}.png")
            break

    if args.using_wandb:
        edges_fname = f"edges.pth"
        exp.save_edges(edges_fname)
        artifact = wandb.Artifact(edges_fname, type="dataset")
        artifact.add_file(edges_fname)
        wandb.log_artifact(artifact)
        os.remove(edges_fname)
        wandb.finish()

    exp.save_subgraph(return_it=True)


if __name__ == "__main__":
    main()