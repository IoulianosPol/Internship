# %% md
< h1 > ACDC
LLaMA
8 - bit
Main
Demo < / h1 >

< p > This
notebook
shows
the
use
of
ACDC
on
the
Induction
task
using
LLaMA - 2 - 7
B in 8 - bit
quantization. < / p >

# %%
# 1. HuggingFace Login (ΑΠΑΡΑΙΤΗΤΟ ΓΙΑ LLaMA)
!pip
install - q
huggingface_hub
from huggingface_hub import login

# ΒΑΛΕ ΤΟ ΔΙΚΟ ΣΟΥ TOKEN ΕΔΩ! (Πρέπει να έχεις κάνει accept το license του LLaMA 2 στο HuggingFace)
login("hf_ΕΔΩ_ΤΟ_TOKEN_ΣΟΥ")

# %%
!rm - rf
ims /
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")

    import subprocess  # to install graphviz dependencies

    command = ['apt-get', 'install', 'graphviz-dev']
    subprocess.run(command, check=True)

    import os  # make images folder

    os.mkdir("ims/")

    from IPython import get_ipython

    ipython = get_ipython()

    ipython.run_line_magic(  # install ACDC
        "pip",
        "install git+https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git@d89f7fa9cbd095202f3940c889cb7c6bf5a9b516",
    )

except Exception as e:
    IN_COLAB = False
    print("Running outside of colab")

    import numpy
    import plotly

    plotly.io.renderers.default = "colab"

    import os

    if not os.path.exists("ims/"):
        os.mkdir("ims/")

    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        print("Running as a notebook")
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
    else:
        print("Running as a script")

# %%
# Download packages (Προσθήκη bitsandbytes & accelerate για 8-bit LLaMA)
!apt - get
install
graphviz - dev - y > / dev / null
!pip
install - q
transformer_lens
einops
kaleido
cmapy
torchtyping
wandb
pygraphviz
bitsandbytes
accelerate

# Download code in collab
!rm - rf
Automatic - Circuit - Discovery
!git
clone
https: // github.com / ArthurConmy / Automatic - Circuit - Discovery.git
%cd
Automatic - Circuit - Discovery
!git
checkout - q
d89f7fa9cbd095202f3940c889cb7c6bf5a9b516
%cd / content /

# %% md
< h2 > Imports and Fixes < / h2 >
# %%
import subprocess
import sys
import importlib
import os
import datetime
import time

# 2. Clear cache
import site

importlib.reload(site)
importlib.invalidate_caches()

# 3. THE ULTIMATE IN-MEMORY BYPASSES
import transformers

transformers.TRANSFORMERS_CACHE = "/tmp/hf_cache"

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

if '/content/Automatic-Circuit-Discovery' not in sys.path:
    sys.path.append('/content/Automatic-Circuit-Discovery')

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

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import HookedTransformer

from acdc.acdc_utils import (
    make_nd_dict,
    reset_network,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
    kl_divergence,
    MatchNLLMetric,
    negative_log_probs
)

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.docstring.utils import AllDataThings

from acdc.induction.utils import (
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.acdc_graphics import show
import argparse

torch.autograd.set_grad_enabled(False)
print("\n Imports Success.")

# %% md
< h2 > Setup
LLaMA - 2 - 7
B(8 - bit) and Task
Data < / h2 >
# %%

os.environ["TRANSFORMERS_USE_FAST"] = "True"


# --- ΑΝΤΙΚΑΤΑΣΤΑΣΗ ΤΗΣ get_model ΓΙΑ ΝΑ ΦΟΡΤΩΝΕΙ LLaMA 8-BIT ---
def get_model(device):
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading {model_name} in 8-bit...")

    # Φόρτωση του μοντέλου σε 8-bit
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Το LLaMA χρειάζεται να του ορίσουμε pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tl_model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
    )

    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    return tl_model


# --- ΕΠΑΝΑΟΡΙΣΜΟΣ ΤΗΣ get_all_induction_things ΜΕΣΑ ΣΤΟ SCRIPT ---
def get_all_induction_things(num_examples, seq_len, device, data_seed=42, metric="kl_div",
                             return_one_element=True) -> AllDataThings:
    tl_model = get_model(device=device)

    validation_data_orig = get_validation_data(device=device)
    mask_orig = get_mask_repeat_candidates(num_examples=None, device=device)
    assert validation_data_orig.shape == mask_orig.shape
    assert seq_len <= validation_data_orig.shape[1] - 1

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

    validation_metric = partial(
        kl_divergence,
        base_model_logprobs=base_val_logprobs,
        mask_repeat_candidates=validation_mask,
        last_seq_element_only=False,
        return_one_element=return_one_element,
    )

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        ),
    }

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=validation_mask,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=test_mask,
        test_patch_data=test_patch_data,
    )


# --- Configuration & Arguments ---
class DummyArgs:
    pass


args = DummyArgs()
args.task = "induction"
args.threshold = 1.0  # ΑΥΞΗΜΕΝΟ THRESHOLD ΓΙΑ ΓΡΗΓΟΡΟΤΕΡΟ PRUNING ΣΤΟ LLAMA
args.zero_ablation = False
args.indices_mode = "reverse"
args.names_mode = "normal"
args.device = "cuda"
args.metric = "kl_div"
args.seed = 1234
args.max_num_epochs = 100000
args.single_step = True  # ΤΟ ΒΑΛΑΜΕ TRUE ΓΙΑ ΝΑ ΚΑΝΕΙ ΕΝΑ ΒΗΜΑ ΜΟΝΟ ΚΑΙ ΝΑ ΜΗΝ ΚΟΛΛΗΣΕΙ
args.abs_value_threshold = False
args.wandb_dir = "/tmp/wandb"
args.wandb_mode = "online"

ONLINE_CACHE_CPU = False
CORRUPTED_CACHE_CPU = False
THRESHOLD = args.threshold
ZERO_ABLATION = args.zero_ablation
USING_WANDB = False  # Κλειστό το Wandb για απλότητα
INDICES_MODE = args.indices_mode
NAMES_MODE = args.names_mode
DEVICE = args.device
RESET_NETWORK = False
SINGLE_STEP = args.single_step

torch.manual_seed(args.seed)

print(f"Task: {args.task} with threshold: {THRESHOLD}...")

# Μειώνουμε πολύ τα examples και το length για να αντέξει η RAM/VRAM με το LLaMA
num_examples = 2
seq_len = 50

things = get_all_induction_things(
    num_examples=num_examples, seq_len=seq_len, device=DEVICE, metric=args.metric
)

print("\nData Success!")

# %% md
< h2 > Setup
ACDC
Experiment < / h2 >
# %%

validation_metric = things.validation_metric
toks_int_values = things.validation_data
toks_int_values_other = things.validation_patch_data
tl_model = things.tl_model

tl_model.reset_hooks()
gc.collect()
torch.cuda.empty_cache()

WANDB_RUN_NAME = "llama_induction_test"
notes = "Testing LLaMA 8-bit ACDC"

exp = TLACDCExperiment(
    model=tl_model,
    threshold=THRESHOLD,
    using_wandb=USING_WANDB,
    wandb_entity_name="test",
    wandb_project_name="acdc",
    wandb_run_name=WANDB_RUN_NAME,
    wandb_group_name="default",
    wandb_notes=notes,
    wandb_dir=args.wandb_dir,
    wandb_mode=args.wandb_mode,
    wandb_config=args,
    zero_ablation=ZERO_ABLATION,
    abs_value_threshold=args.abs_value_threshold,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=validation_metric,
    second_metric=None,
    verbose=True,
    indices_mode=INDICES_MODE,
    names_mode=NAMES_MODE,
    corrupted_cache_cpu=CORRUPTED_CACHE_CPU,
    hook_verbose=False,
    online_cache_cpu=ONLINE_CACHE_CPU,
    add_sender_hooks=True,
    use_pos_embed=False,
    add_receiver_hooks=False,
    remove_redundant=False,
    show_full_index=False,
)

# %% md
< h2 > View
Prompts and Induction
Targets < / h2 >
# %%

val_data = things.validation_data
mask = things.validation_mask

for prompt_idx in range(len(val_data)):
    prompt_tokens = val_data[prompt_idx]
    prompt_text = tl_model.to_string(prompt_tokens)

    print("\n" + "=" * 60)
    print(f"--- Real Text (Prompt {prompt_idx}) ---")
    print(prompt_text)

    print(f"\n--- Induction Targets (Mask) for Prompt {prompt_idx} ---")

    induction_count = 0
    for i, (token_id, is_induction_target) in enumerate(zip(prompt_tokens, mask[prompt_idx])):
        if is_induction_target:
            print(f"Position {i}: The model was expected to predict token '{tl_model.to_string(token_id)}'")
            induction_count += 1

    if induction_count == 0:
        print("No repeating induction patterns were found in this prompt.")

# %% md
< h2 > Run
steps
of
ACDC < / h2 >
# %%

exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
last_edge_count = exp.count_no_edges()

for i in range(args.max_num_epochs):
    print(f"\n--- Starting Step {i + 1} ---")
    exp.step(testing=False)

    current_edge_count = exp.count_no_edges()

    if current_edge_count < last_edge_count:
        print(f"Edge removed! New set: {current_edge_count}")
        fname = f"ims/img_pruned_{i + 1}.png"

        # Λόγω τεράστιου μεγέθους LLaMA, ίσως η ζωγραφιά του γράφου να κρασάρει,
        # Οπότε το βάζουμε σε try/except block
        try:
            show(exp.corr, fname=fname, show_full_index=False)
            if IN_COLAB or ipython is not None:
                display(Image(fname))
        except Exception as e:
            print(f"Could not render graph (too large?): {e}")

        last_edge_count = current_edge_count

    print(i, "-" * 50)
    print(f"Edges remaining: {current_edge_count}")

    if i == 0:
        exp.save_edges("edges.pkl")

    if exp.current_node is None or SINGLE_STEP:
        print("\nStopping ACDC loop (Single Step is True or Graph Finished).")
        try:
            show(exp.corr, f"ims/ACDC_img_{exp_time}.png")
        except:
            pass
        break