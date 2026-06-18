#%% md
# <h1>ACDC Main Demo</h1>
# 
# <p>This notebook (which doubles as a script) shows several use cases of ACDC</p>
# 
# <p>The codebase is built on top of https://github.com/neelnanda-io/TransformerLens (source version)</p>
# 
# <h3>Setup:</h3>
# <p>Janky code to do different setup when run in a Colab notebook vs VSCode (adapted from e.g <a href="https://github.com/neelnanda-io/TransformerLens/blob/5c89b7583e73ce96db5e46ef86a14b15f303dde6/demos/Activation_Patching_in_TL_Demo.ipynb">this notebook</a>)</p>
#%%
!rm -rf ims/
#%%
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")

    import subprocess # to install graphviz dependencies
    command = ['apt-get', 'install', 'graphviz-dev']
    subprocess.run(command, check=True)

    import os # make images folder
    os.mkdir("ims/")

    from IPython import get_ipython
    ipython = get_ipython()

    ipython.run_line_magic( # install ACDC
        "pip",
        "install git+https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git@d89f7fa9cbd095202f3940c889cb7c6bf5a9b516",
    )

except Exception as e:
    IN_COLAB = False
    print("Running outside of colab")

    import numpy # crucial to not get cursed error
    import plotly

    plotly.io.renderers.default = "colab"  # added by Arthur so running as a .py notebook with #%% generates .ipynb notebooks that display in colab
    # disable this option when developing rather than generating notebook outputs

    import os # make images folder
    if not os.path.exists("ims/"):
        os.mkdir("ims/")

    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        print("Running as a notebook")
        ipython.run_line_magic("load_ext", "autoreload")  # type: ignore
        ipython.run_line_magic("autoreload", "2")  # type: ignore
    else:
        print("Running as a script")
#%%
# Download packages
!apt-get install graphviz-dev -y > /dev/null
!pip install -q bitsandbytes accelerate transformer_lens huggingface_hub einops kaleido cmapy torchtyping wandb pygraphviz

# Download code in collab
!rm -rf Automatic-Circuit-Discovery #
!git clone https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git
%cd Automatic-Circuit-Discovery
!git checkout -q d89f7fa9cbd095202f3940c889cb7c6bf5a9b516
%cd /content/
#%% md
# <h2>Imports etc</h2>
#%%
import subprocess
import sys
import importlib
import os
import datetime
import time
# 1. Installation
print("Installing packages")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformer-lens==1.5.0", "einops", "kaleido", "cmapy", "wandb", "pygraphviz"
], check=True)

# 2. Clear cache
import site
importlib.reload(site)
importlib.invalidate_caches()

# 3. THE ULTIMATE IN-MEMORY BYPASSES

# A. HuggingFace Cache Fix (Protect transformer_lens)
import transformers
transformers.TRANSFORMERS_CACHE = "/tmp/hf_cache"

# B. Transformer Lens Rename Fix
import transformer_lens
sys.modules['transformer_lens.HookedTransformerConfig'] = transformer_lens
sys.modules['transformer_lens.HookedTransformer'] = transformer_lens
sys.modules['transformer_lens.ActivationCache'] = transformer_lens
sys.modules['transformer_lens.FactoredMatrix'] = transformer_lens

# C. Torchtyping & Typeguard Fix
import types
# Turnoff typeguard
dummy_typeguard = types.ModuleType("typeguard")
dummy_typeguard.typechecked = lambda func: func  # Return function
sys.modules["typeguard"] = dummy_typeguard

# Fake torchtyping
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
import argparse

torch.autograd.set_grad_enabled(False)
print("\n Success.")

# GPU check
#assert torch.cuda.is_available(), "GPU was not found ! "

#%%
import dataclasses
from functools import partial
from acdc.docstring.utils import AllDataThings
import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Dict, Callable
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
import warnings
import networkx as nx
from acdc.acdc_utils import (
    MatchNLLMetric,
    make_nd_dict,
    shuffle_tensor,
)

from acdc.TLACDCEdge import (
    TorchIndex,
    Edge,
    EdgeType,
)
from transformer_lens import HookedTransformer
from acdc.acdc_utils import kl_divergence, negative_log_probs

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

def get_good_induction_candidates(num_examples=None, seq_len=None, device=None):
    good_induction_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
    )
    good_induction_candidates = torch.load(good_induction_candidates_fname, map_location=device)

    if num_examples is None:
        return good_induction_candidates
    else:
        return good_induction_candidates[:num_examples][:seq_len]

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


def get_all_induction_things(num_examples, seq_len, device, data_seed=42, metric="kl_div", return_one_element=True) -> AllDataThings:

    tl_model = get_model(device=device)

    validation_data_orig = get_validation_data(device=device)

    vocab_size = tl_model.cfg.d_vocab
    safe_token = tl_model.tokenizer.eos_token_id
    validation_data_orig[validation_data_orig >= vocab_size] = safe_token

    mask_orig = get_mask_repeat_candidates(num_examples=None, device=device)
    assert validation_data_orig.shape == mask_orig.shape

    assert seq_len <= validation_data_orig.shape[1]-1

    validation_slice = slice(0, num_examples)
    validation_data = validation_data_orig[validation_slice, :seq_len].contiguous()
    validation_labels = validation_data_orig[validation_slice, 1:seq_len+1].contiguous()
    validation_mask = mask_orig[validation_slice, :seq_len].contiguous()

    validation_patch_data = shuffle_tensor(validation_data, seed=data_seed).contiguous()

    test_slice = slice(num_examples, num_examples*2)
    test_data = validation_data_orig[test_slice, :seq_len].contiguous()
    test_labels = validation_data_orig[test_slice, 1:seq_len+1].contiguous()
    test_mask = mask_orig[test_slice, :seq_len].contiguous()

    test_patch_data = shuffle_tensor(test_data, seed=data_seed).contiguous()

    with torch.no_grad():
        base_val_logprobs = F.log_softmax(tl_model(validation_data), dim=-1).detach()
        base_test_logprobs = F.log_softmax(tl_model(test_data), dim=-1).detach()

    if metric == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_val_logprobs,
            mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
            return_one_element=return_one_element,
        )
    elif metric == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
        )
    elif metric == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels, base_model_logprobs=base_val_logprobs, mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
        )
    else:
        raise ValueError(f"Unknown metric {metric}")

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        ),
        "match_nll": MatchNLLMetric(
            labels=test_labels, base_model_logprobs=base_test_logprobs, mask_repeat_candidates=test_mask,
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


def one_item_per_batch(toks_int_values, toks_int_values_other, mask_rep, base_model_logprobs, kl_take_mean=True):
    end_positions = []
    batch_size, seq_len = toks_int_values.shape
    new_tensors = []

    toks_int_values_other_batch_list = []
    new_base_model_logprobs_list = []

    for i in range(batch_size):
        for j in range(seq_len - 1):
            if mask_rep[i, j]:
                end_positions.append(j)
                new_tensors.append(toks_int_values[i].cpu().clone())
                toks_int_values_other_batch_list.append(toks_int_values_other[i].cpu().clone())
                new_base_model_logprobs_list.append(base_model_logprobs[i].cpu().clone())

    toks_int_values_other_batch = torch.stack(toks_int_values_other_batch_list).to(toks_int_values.device).clone()
    return_tensor = torch.stack(new_tensors).to(toks_int_values.device).clone()
    end_positions_tensor = torch.tensor(end_positions).long()

    new_base_model_logprobs = torch.stack(new_base_model_logprobs_list)[torch.arange(len(end_positions_tensor)), end_positions_tensor].to(toks_int_values.device).clone()
    metric = partial(
        kl_divergence,
        base_model_logprobs=new_base_model_logprobs,
        end_positions=end_positions_tensor,
        mask_repeat_candidates=None,
        last_seq_element_only=False,
        return_one_element=False
    )

    return return_tensor, toks_int_values_other_batch, end_positions_tensor, metric
#%% md
# <h2>Setup Task</h2>
#%%
import os
import argparse
import torch
import transformers
from IPython import get_ipython

os.environ["TRANSFORMERS_USE_FAST"] = "True"

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

ipython = get_ipython()
parser = argparse.ArgumentParser(description="Used to launch ACDC runs. Only task and threshold are required")

task_choices = ['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan']
parser.add_argument('--task', type=str, required=True, choices=task_choices, help=f'Choose a task from the available options: {task_choices}')
parser.add_argument('--threshold', type=float, required=True, help='Value for THRESHOLD')
parser.add_argument('--first-cache-cpu', type=str, required=False, default="True", help='Value for FIRST_CACHE_CPU')
parser.add_argument('--second-cache-cpu', type=str, required=False, default="True", help='Value for SECOND_CACHE_CPU')
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument('--using-wandb', action='store_true', help='Use wandb')
parser.add_argument('--wandb-entity-name', type=str, required=False, default="remix_school-of-rock")
parser.add_argument('--wandb-group-name', type=str, required=False, default="default")
parser.add_argument('--wandb-project-name', type=str, required=False, default="acdc")
parser.add_argument('--wandb-run-name', type=str, required=False, default=None)
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument('--indices-mode', type=str, default="normal")
parser.add_argument('--names-mode', type=str, default="normal")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--reset-network', type=int, default=0)
parser.add_argument('--metric', type=str, default="kl_div")
parser.add_argument('--torch-num-threads', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument("--max-num-epochs",type=int, default=100_000)
parser.add_argument('--single-step', action='store_true')
parser.add_argument("--abs-value-threshold", action='store_true')

if ipython is not None:
    args = parser.parse_args(
        [line.strip() for line in r"""--task=induction\
--threshold=0.5623\
--zero-ablation\
--indices-mode=reverse\
--first-cache-cpu=True\
--second-cache-cpu=True\
--max-num-epochs=100000""".split("\\\n")]
    )
else:
    args = parser.parse_args()

if args.torch_num_threads > 0:
    torch.set_num_threads(args.torch_num_threads)
torch.manual_seed(args.seed)

TASK = args.task
if args.first_cache_cpu is None:
    ONLINE_CACHE_CPU = True
elif args.first_cache_cpu.lower() == "false":
    ONLINE_CACHE_CPU = False
elif args.first_cache_cpu.lower() == "true":
    ONLINE_CACHE_CPU = True
else:
    raise ValueError(f"first_cache_cpu must be either True or False")

if args.second_cache_cpu is None:
    CORRUPTED_CACHE_CPU = True
elif args.second_cache_cpu.lower() == "false":
    CORRUPTED_CACHE_CPU = False
elif args.second_cache_cpu.lower() == "true":
    CORRUPTED_CACHE_CPU = True
else:
    raise ValueError(f"second_cache_cpu must be either True or False")

THRESHOLD = args.threshold
ZERO_ABLATION = False if args.zero_ablation else False
USING_WANDB = True if args.using_wandb else False
WANDB_ENTITY_NAME = args.wandb_entity_name
WANDB_PROJECT_NAME = args.wandb_project_name
WANDB_RUN_NAME = args.wandb_run_name
WANDB_GROUP_NAME = args.wandb_group_name
INDICES_MODE = args.indices_mode
NAMES_MODE = args.names_mode
DEVICE = args.device
RESET_NETWORK = args.reset_network
SINGLE_STEP = True if args.single_step else False

print(f"Task: {TASK} with threshold: {THRESHOLD}...")

second_metric = None
use_pos_embed = TASK.startswith("tracr")

if TASK == "ioi":
    num_examples = 40
    things = get_all_ioi_things(
        num_examples=num_examples, device=DEVICE, metric_name=args.metric
    )
elif TASK == "tracr-reverse":
    num_examples = 6
    things = get_all_tracr_things(
        task="reverse",
        metric_name=args.metric,
        num_examples=num_examples,
        device=DEVICE,
    )
elif TASK == "tracr-proportion":
    num_examples = 50
    things = get_all_tracr_things(
        task="proportion",
        metric_name=args.metric,
        num_examples=num_examples,
        device=DEVICE,
    )
elif TASK == "induction":
    num_examples = 2
    seq_len = 300
    things = get_all_induction_things(
        num_examples=num_examples, seq_len=seq_len, device=DEVICE, metric=args.metric
    )
elif TASK == "docstring":
    num_examples = 50
    seq_len = 41
    things = get_all_docstring_things(
        num_examples=num_examples,
        seq_len=seq_len,
        device=DEVICE,
        metric_name=args.metric,
        correct_incorrect_wandb=True,
    )
elif TASK == "greaterthan":
    num_examples = 100
    things = get_all_greaterthan_things(
        num_examples=num_examples, metric_name=args.metric, device=DEVICE
    )
else:
    raise ValueError(f"Unknown task {TASK}")

print("\nSuccess!")
#%% md
# <p> Let's define the four most important objects for ACDC experiments:
#%%

validation_metric = things.validation_metric # metric we use (e.g KL divergence)
toks_int_values = things.validation_data # clean data x_i
toks_int_values_other = things.validation_patch_data # corrupted data x_i'
tl_model = things.tl_model # transformerlens model

if RESET_NETWORK:
    reset_network(TASK, DEVICE, tl_model)

#%% md
# <h2>Setup ACDC Experiment</h2>
#%%
# Make notes for potential wandb run
try:
    with open(__file__, "r") as f:
        notes = f.read()
except:
    notes = "No notes generated, expected when running in an .ipynb file"

tl_model.reset_hooks()

# Save some mem
gc.collect()
torch.cuda.empty_cache()

# Setup wandb if needed
if WANDB_RUN_NAME is None or IPython.get_ipython() is not None:
    WANDB_RUN_NAME = f"{ct()}{'_randomindices' if INDICES_MODE=='random' else ''}_{THRESHOLD}{'_zero' if ZERO_ABLATION else ''}"
else:
    assert WANDB_RUN_NAME is not None, "I want named runs, always"

tl_model.reset_hooks()
exp = TLACDCExperiment(
    model=tl_model,
    threshold=THRESHOLD,
    using_wandb=USING_WANDB,
    wandb_entity_name=WANDB_ENTITY_NAME,
    wandb_project_name=WANDB_PROJECT_NAME,
    wandb_run_name=WANDB_RUN_NAME,
    wandb_group_name=WANDB_GROUP_NAME,
    wandb_notes=notes,
    wandb_dir=args.wandb_dir,
    wandb_mode=args.wandb_mode,
    wandb_config=args,
    zero_ablation=ZERO_ABLATION,
    abs_value_threshold=args.abs_value_threshold,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=validation_metric,
    second_metric=second_metric,
    verbose=True,
    indices_mode=INDICES_MODE,
    names_mode=NAMES_MODE,
    corrupted_cache_cpu=CORRUPTED_CACHE_CPU,
    hook_verbose=False,
    online_cache_cpu=ONLINE_CACHE_CPU,
    add_sender_hooks=True,
    use_pos_embed=use_pos_embed,
    add_receiver_hooks=False,
    remove_redundant=False,
    show_full_index=use_pos_embed,
)

#%%
# Instead of reloading the data, we extract it directly
# from the 'things' object used in the ACDC Experiment
val_data = things.validation_data
mask = things.validation_mask

# Loop through all 10 (or the specified number of) prompts
for prompt_idx in range(len(val_data)):
    prompt_tokens = val_data[prompt_idx]
    prompt_text = tl_model.to_string(prompt_tokens)

    print("\n" + "="*60)
    print(f"--- Real Text (Prompt {prompt_idx}) ---")
    print(prompt_text)

    print(f"\n--- Induction Targets (Mask) for Prompt {prompt_idx} ---")

    # Counter to track how many induction tokens were found
    induction_count = 0
    for i, (token_id, is_induction_target) in enumerate(zip(prompt_tokens, mask[prompt_idx])):
        if is_induction_target:
            print(f"Position {i}: The model was expected to predict token '{tl_model.to_string(token_id)}'")
            induction_count += 1

    if induction_count == 0:
        print("No repeating induction patterns were found in this prompt.")
#%%
# Model config
print(tl_model.cfg)
#%%
from IPython.display import Image, display
from acdc.acdc_graphics import show

print("Initial Edges:", exp.count_no_edges())
all_nodes = [
    node
    for receiver_dict in exp.corr.graph.values()
    for node in receiver_dict.values()
]

print("Initial Nodes:", len(all_nodes))
print("-" * 50)
show(
    exp.corr,
    "ims/initial_full_network.png",
    show_full_index=False,
)

display(Image("ims/initial_full_network.png"))
print("-" * 50)
print(exp.corr.nodes())
print("Initial connections")
for edge_tuple, edge in exp.corr.all_edges().items():
    if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
        receiver_name, receiver_idx, sender_name, sender_idx = edge_tuple

        rec_str = f"{receiver_name} {receiver_idx.hashable_tuple}"
        send_str = f"{sender_name} {sender_idx.hashable_tuple}"

        print(f"From: {send_str}  --->  To: {rec_str}")
#%%
import networkx as nx

G = nx.DiGraph()

for (receiver_name, receiver_index, sender_name, sender_index), edge in exp.corr.all_edges().items():
    if not edge.present:
        continue

    src = f"{sender_name}{sender_index}"
    dst = f"{receiver_name}{receiver_index}"

    G.add_edge(src, dst)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2000,
    font_size=8,
    arrows=True
)

plt.show()
#%% md
# <h2>Run steps of ACDC: iterate over a NODE in the model's computational graph</h2>
# <p>WARNING! This will take a few minutes to run, but there should be rolling nice pictures too : )</p>
#%%

import datetime
exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# for i in range(args.max_num_epochs):
#     exp.step(testing=False)

#     show(
#         exp.corr,
#         f"ims/img_new_{i+1}.png",
#         show_full_index=False,
#     )

#     if IN_COLAB or ipython is not None:
#         # so long as we're not running this as a script, show the image!
#         display(Image(f"ims/img_new_{i+1}.png"))

#     print(i, "-" * 50)
#     print(exp.count_no_edges())

#     if i == 0:
#         exp.save_edges("edges.pkl")

#     if exp.current_node is None or SINGLE_STEP:
#         show(
#             exp.corr,
#             f"ims/ACDC_img_{exp_time}.png",

#         )
#         break

# exp.save_edges("another_final_edges.pkl")

#DISPLAY GRAPH WHEN EDGE IS REMOVED
last_edge_count = exp.count_no_edges()

for i in range(args.max_num_epochs):
    exp.step(testing=False)

    current_edge_count = exp.count_no_edges()

    if current_edge_count < last_edge_count:
        print(f"Edge removed! New set: {current_edge_count}")

        fname = f"ims/img_pruned_{i+1}.png"
        show(exp.corr, fname=fname, show_full_index=False)

        if IN_COLAB or ipython is not None:
            display(Image(fname))

        last_edge_count = current_edge_count

    print(i, "-" * 50)
    print(f"Edges remaining: {current_edge_count}")

    if i == 0:
        exp.save_edges("edges.pkl")

    if exp.current_node is None or SINGLE_STEP:
        show(exp.corr, f"ims/ACDC_img_{exp_time}.png")
        break

if USING_WANDB:
    edges_fname = f"edges.pth"
    exp.save_edges(edges_fname)
    artifact = wandb.Artifact(edges_fname, type="dataset")
    artifact.add_file(edges_fname)
    wandb.log_artifact(artifact)
    os.remove(edges_fname)
    wandb.finish()

#%% md
# <h2>Save the final subgraph of the model</h2>
# <p>There are more than `exp.count_no_edges()` here because we include some "placeholder" edges needed to make ACDC work that don't actually matter</p>
# <p>Also note that the final image has more than 12 edges, because the edges from a0.0_q and a0.0_k are not connected to the input</p>
# <p>We recover minimal induction machinery! `embed -> a0.0_v -> a1.6k`</p>
#%%
exp.save_subgraph(
    return_it=True,
)