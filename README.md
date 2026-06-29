# Automated Circuit Discovery Notebooks

Welcome to the **Automated Circuit Discovery (ACDC)** notebooks repository!

This repository contains Google Colab notebooks and scripts for running **Automated Circuit Discovery (ACDC)** experiments on Transformer models. It allows you to configure different tasks, adjust discovery thresholds, and optionally generate the initial computational graph for visualization and analysis.

---

## Overview

Current features include:

-  Google Colab compatible notebooks
-  GPT-2 circuit discovery experiments
-  Configurable tasks
-  Adjustable discovery thresholds
-  Optional initial graph generation
-  Easy experimentation and reproducibility

---

## Running the Notebooks

Open the desired notebook in **Google Colab** and execute the cells sequentially (e.g acdc_my_demo_docstring.ipynb )

For GPT-2 experiments, run:

```bash
python acdc/acdc_main_gpt2.py
```

---

##  Configuration

The main experiment configuration is located in:

```text
acdc/acdc_main_gpt2.py
```

Inside this file you can modify:

- **Task** – choose the task you want to run.
- **Thresholds** – adjust the pruning/discovery thresholds.
- **Model parameters** – customize experiment settings if necessary.

---

##  Generating the Initial Graph

If you would like to generate the **initial computational graph**, simply locate the corresponding section inside:

```text
acdc/acdc_main_gpt2.py
```

and **uncomment the provided code**.

After enabling it, rerun the notebook to generate the graph before the circuit discovery process.

---

##  Project Structure

```text
.
├── acdc/
│   ├── acdc_main_gpt2.py
│   ├── notebooks/s
```

---

## 🛠 Typical Workflow

1. Open the notebook in **Google Colab**.
2. Install all required dependencies.
3. Configure the desired task.
4. Set the thresholds.
5. Run the notebook.

---

## Notes

- Different tasks may require different threshold values.
- Lower thresholds generally produce larger circuits.
- Generating the initial graph is optional but recommended for visualization and debugging.
- For reproducible experiments, keep track of the selected task and threshold values.

---

## Example for GPT2

```bash
# Run GPT-2 ACDC
python acdc/acdc_main_gpt2.py
```

Then edit `acdc_main_gpt2.py` to:

- change the task
- modify the thresholds
- uncomment the initial graph generation code (if needed)

and rerun the acdc_main_gpt2.py.

---

Happy circuit discovering!
By Ioulianos Polyzos
