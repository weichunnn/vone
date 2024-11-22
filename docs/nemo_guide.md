[Skip to main content](#main-content)

Back to top  

 Ctrl+K

[![NVIDIA NeMo Framework User Guide - Home](../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg)

NVIDIA NeMo Framework User Guide

](../index.html)

> Important
> 
> You are viewing the NeMo 2.0 documentation. This release introduces significant changes to the API and a new library, [NeMo Run](https://github.com/NVIDIA/NeMo-Run). We are currently porting all features from NeMo 1.0 to 2.0. For documentation on previous versions or features not yet available in 2.0, please refer to the [NeMo 24.07 documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.07/overview.html).

# Quickstart with NeMo-Run[#](#quickstart-with-nemo-run "Link to this heading")

This is an introduction to running any of the supported [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run). In this tutorial, we will take a pretraining and finetuning recipe and try to run it locally, as well as remotely, on a Slurm-based cluster. Let’s get started.

Please go through the [NeMo-Run README](https://github.com/NVIDIA/NeMo-Run/blob/main/README.md) to get a high-level overview of NeMo-Run.

## Minimum Requirements[#](#minimum-requirements "Link to this heading")

This tutorial requires a minimum of 1 NVIDIA GPU with atleast 48GB memory for [finetuning](#finetuning-quickstart), and 2 NVIDIA GPUs with atleast 48GB memory each for [pretraining](#pretraining-quickstart) (although it can be done on a single GPU or GPUs with lesser memory by decreasing the model size further). Each section can be followed individually based on your needs. You will also need to run this tutorial inside the [NeMo container](https://docs.nvidia.com/nemo-framework/user-guide/latest/getting-started.html#get-access-to-nemo-framework) with the `dev` tag.

## Pretraining[#](#pretraining "Link to this heading")

For the purposes of this pretraining quickstart, we will use a relatively small model. We will begin with the [Nemotron 3 4B pretraining recipe](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/nemotron3_4b.py), and go through the steps required to configure and launch pretraining.

As mentioned in the requirements, this tutorial was run on a node with 2 GPUs (each RTX 5880 with 48GB memory). If you intend to run on just 1 GPU or GPUs with lesser memory, please change the configuration to match your host. For example, you can reduce num\_layers or hidden\_size in the model config to make it fit on a single GPU.

### Set up the Prerequisites[#](#set-up-the-prerequisites "Link to this heading")

Run the following commands to set up your workspace and files:

\# Check GPU access
nvidia-smi

\# Create and go to workspace
mkdir \-p /workspace/nemo-run
cd /workspace/nemo-run

\# Create a python file to run pre-training
touch nemotron\_pretraining.py

### Configure the Recipe[#](#configure-the-recipe "Link to this heading")

Configure the recipe inside nemotron\_pretraining.py:

import nemo\_run as run

from nemo.collections import llm

def configure\_recipe(nodes: int \= 1, gpus\_per\_node: int \= 2):
    recipe \= llm.nemotron3\_4b.pretrain\_recipe(
        dir\="/checkpoints/nemotron", \# Path to store checkpoints
        name\="nemotron\_pretraining",
        tensor\_parallelism\=2,
        num\_nodes\=nodes,
        num\_gpus\_per\_node\=gpus\_per\_node,
        max\_steps\=100, \# Setting a small value for the quickstart
    )

    \# Add overrides here

    return recipe

Here, the recipe variable holds a configured run.Partial object. Please read about the configuration system in NeMo-Run [here](https://github.com/NVIDIA/NeMo-Run/blob/main/docs/source/guides/configuration.md) for more details. For those familiar with the NeMo 1.0-style YAML configuration, this recipe is just a Pythonic version of a YAML config file for pretraining.

#### Override attributes[#](#override-attributes "Link to this heading")

You can set overrides on its attributes like normal Python objects. So, if want to change the val\_check\_interval, you can override it after defining your recipe by setting:

recipe.trainer.val\_check\_interval \= 100

Note

An important thing to remember is that you are only configuring your task at this stage; the underlying code is not being executed at this time.

#### Swap Recipes[#](#swap-recipes "Link to this heading")

The recipes in NeMo 2.0 are easily swappable. For instance, if you want to swap the NeMotron recipe with a Llama 3 recipe, you can simply run the following command:

recipe \= llm.llama3\_8b.pretrain\_recipe(
    dir\="/checkpoints/llama3", \# Path to store checkpoints
    name\="llama3\_pretraining",
    num\_nodes\=nodes,
    num\_gpus\_per\_node\=gpus\_per\_node,
)

Once you have the final recipe configured, you are ready to move to the execution stage.

### Execute Locally[#](#execute-locally "Link to this heading")

1.  First, we will execute locally using torchrun. In order to do that, we will define a LocalExecutor as shown:
    

def local\_executor\_torchrun(nodes: int \= 1, devices: int \= 2) \-> run.LocalExecutor:
    \# Env vars for jobs are configured here
    env\_vars \= {
        "TORCH\_NCCL\_AVOID\_RECORD\_STREAMS": "1",
        "NCCL\_NVLS\_ENABLE": "0",
        "NVTE\_DP\_AMAX\_REDUCE\_INTERVAL": "0",
        "NVTE\_ASYNC\_AMAX\_REDUCTION": "1",
        "NVTE\_FUSED\_ATTN": "0",
    }

    executor \= run.LocalExecutor(ntasks\_per\_node\=devices, launcher\="torchrun", env\_vars\=env\_vars)

    return executor

To find out more about NeMo-Run executors, see the [execution](https://github.com/NVIDIA/NeMo-Run/blob/main/docs/source/guides/execution.md) guide.

2.  Next, we will combine the recipe and executor to launch the pretraining run:
    

def run\_pretraining():
    recipe \= configure\_recipe()
    executor \= local\_executor\_torchrun(nodes\=recipe.trainer.num\_nodes, devices\=recipe.trainer.devices)

    run.run(recipe, executor\=executor)

\# Wrap the call in an if \_\_name\_\_ == "\_\_main\_\_": block to work with Python's multiprocessing module.
if \_\_name\_\_ \== "\_\_main\_\_":
    run\_pretraining()

The full code for nemotron\_pretraining.py looks like:

import nemo\_run as run

from nemo.collections import llm

def configure\_recipe(nodes: int \= 1, gpus\_per\_node: int \= 2):
    recipe \= llm.nemotron3\_4b.pretrain\_recipe(
        dir\="/checkpoints/nemotron", \# Path to store checkpoints
        name\="nemotron\_pretraining",
        tensor\_parallelism\=2,
        num\_nodes\=nodes,
        num\_gpus\_per\_node\=gpus\_per\_node,
        max\_steps\=100, \# Setting a small value for the quickstart
    )

    recipe.trainer.val\_check\_interval \= 100
    return recipe

def local\_executor\_torchrun(nodes: int \= 1, devices: int \= 2) \-> run.LocalExecutor:
    \# Env vars for jobs are configured here
    env\_vars \= {
        "TORCH\_NCCL\_AVOID\_RECORD\_STREAMS": "1",
        "NCCL\_NVLS\_ENABLE": "0",
        "NVTE\_DP\_AMAX\_REDUCE\_INTERVAL": "0",
        "NVTE\_ASYNC\_AMAX\_REDUCTION": "1",
        "NVTE\_FUSED\_ATTN": "0",
    }

    executor \= run.LocalExecutor(ntasks\_per\_node\=devices, launcher\="torchrun", env\_vars\=env\_vars)

    return executor

def run\_pretraining():
    recipe \= configure\_recipe()
    executor \= local\_executor\_torchrun(nodes\=recipe.trainer.num\_nodes, devices\=recipe.trainer.devices)

    run.run(recipe, executor\=executor)

\# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if \_\_name\_\_ \== "\_\_main\_\_":
    run\_pretraining()

You can run the file using just:

python nemotron\_pretraining.py

Here’s a recording showing all the steps above leading up to the start of pretraining:

#### Change the number of GPUs[#](#change-the-number-of-gpus "Link to this heading")

Let’s see how we can change the configuration to run on just 1 GPU instead of 2. All you need to do is change the configuration in `run_pretraining`, as shown below:

def run\_pretraining():
    recipe \= configure\_recipe()
    executor \= local\_executor\_torchrun(nodes\=recipe.trainer.num\_nodes, devices\=recipe.trainer.devices)

    \# Change to 1 GPU

    \# Change executor params
    executor.ntasks\_per\_node \= 1
    executor.env\_vars\["CUDA\_VISIBLE\_DEVICES"\] \= "0"

    \# Change recipe params

    \# The default number of layers comes from the recipe in nemo where num\_layers is 32
    \# Ref: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/model/nemotron.py
    \# To run on 1 GPU without TP, we can reduce the number of layers to 8 by setting recipe.model.config.num\_layers = 8
    recipe.model.config.num\_layers \= 8
    \# We also need to set TP to 1, since we had used 2 for 2 GPUs.
    recipe.trainer.strategy.tensor\_model\_parallel\_size \= 1
    \# Lastly, we need to set devices to 1 in the trainer.
    recipe.trainer.devices \= 1

    run.run(recipe, executor\=executor)

### Execute on a Slurm Cluster[#](#execute-on-a-slurm-cluster "Link to this heading")

One of the benefits of NeMo-Run is to allow you to easily scale from local to remote slurm-based clusters. Next, let’s see how we can launch the same pretraining recipe on a Slurm cluster.

1.  First, we’ll define a [slurm executor](https://github.com/NVIDIA/NeMo-Run/blob/main/src/nemo_run/core/execution/slurm.py):
    

def slurm\_executor(
    user: str,
    host: str,
    remote\_job\_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    time: str \= "01:00:00",
    custom\_mounts: Optional\[list\[str\]\] \= None,
    custom\_env\_vars: Optional\[dict\[str, str\]\] \= None,
    container\_image: str \= "nvcr.io/nvidia/nemo:dev",
    retries: int \= 0,
) \-> run.SlurmExecutor:
    if not (user and host and remote\_job\_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote\_job\_dir, account, partition, nodes and devices args for using this function."
        )

    mounts \= \[\]
    \# Custom mounts are defined here.
    if custom\_mounts:
        mounts.extend(custom\_mounts)

    \# Env vars for jobs are configured here
    env\_vars \= {
        "TRANSFORMERS\_OFFLINE": "1",
        "TORCH\_NCCL\_AVOID\_RECORD\_STREAMS": "1",
        "NCCL\_NVLS\_ENABLE": "0",
        "NVTE\_DP\_AMAX\_REDUCE\_INTERVAL": "0",
        "NVTE\_ASYNC\_AMAX\_REDUCTION": "1",
        "NVTE\_FUSED\_ATTN": "0",
    }
    if custom\_env\_vars:
        env\_vars |= custom\_env\_vars

    \# This defines the slurm executor.
    \# We connect to the executor via the tunnel defined by user, host and remote\_job\_dir.
    executor \= run.SlurmExecutor(
        account\=account,
        partition\=partition,
        tunnel\=run.SSHTunnel(
            user\=user,
            host\=host,
            job\_dir\=remote\_job\_dir, \# This is where the results of the run will be stored by default.
            \# identity="/path/to/identity/file" OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
        ),
        nodes\=nodes,
        ntasks\_per\_node\=devices,
        gpus\_per\_node\=devices,
        mem\="0",
        exclusive\=True,
        gres\="gpu:8",
        packager\=run.Packager(),
    )

    executor.container\_image \= container\_image
    executor.container\_mounts \= mounts
    executor.env\_vars \= env\_vars
    executor.retries \= retries
    executor.time \= time

    return executor

2.  Next, you can just replace the local executor with the slurm executor, like below:
    

def run\_pretraining\_with\_slurm():
    recipe \= configure\_recipe(nodes\=1, gpus\_per\_node\=8)
    executor \= slurm\_executor(
        user\="", \# TODO: Set the username you want to use
        host\="", \# TODO: Set the host of your cluster
        remote\_job\_dir\="", \# TODO: Set the directory on the cluster where you want to save results
        account\="", \# TODO: Set the account for your cluster
        partition\="", \# TODO: Set the partition for your cluster
        container\_image\="", \# TODO: Set the container image you want to use for your job
        \# container\_mounts=\[\], TODO: Set any custom mounts
        \# custom\_env\_vars={}, TODO: Set any custom env vars
        nodes\=recipe.trainer.num\_nodes,
        devices\=recipe.trainer.devices,
    )

    run.run(recipe, executor\=executor, detach\=True)

3.  Finally, you can run it as follows:
    

if \_\_name\_\_ \== "\_\_main\_\_":
    run\_pretraining\_with\_slurm()

python nemotron\_pretraining.py

Since we have set detach=True, the process will exit after scheduling the job on the cluster with information about directories and commands to manage the run/experiment.

## Finetuning[#](#finetuning "Link to this heading")

One of the main benefits of NeMo-Run is that it decouples configuration and execution. So we can reuse predefined executors and just change the recipe. For the purpose of this tutorial, we will include the executor definition so that this section can be followed independently.

### Set up the Prerequisites[#](#id1 "Link to this heading")

Run the following commands to set up your Hugginface token for automatic conversion of model from huggingface.

mkdir \-p /tokens

\# Fetch Huggingface token and export it.
\# See https://huggingface.co/docs/hub/en/security-tokens for instructions.
export HF\_TOKEN\="hf\_your\_token" \# Change this to your Huggingface token

\# Save token to /tokens/huggingface
echo "$HF\_TOKEN" \> /tokens/huggingface

### Configure the Recipe[#](#id2 "Link to this heading")

In this quickstart, we will finetune a Llama 3 8B model from huggingface on a single GPU. In order to do this, we need two steps:

1.  Convert the checkpoint from Hugginface to NeMo
    
2.  Run finetuning using the converted checkpoint from 1.
    

We will do this using a [NeMo-Run experiment](https://github.com/NVIDIA/NeMo-Run/blob/main/src/nemo_run/run/experiment.py#L66), which allows you to define these two tasks and execute them sequentially in an easy way. We will do this inside a new file `nemotron_finetuning.py` in the same directory. We will use the [Llama3 8b finetuning recipe](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/llama3_8b.py#L254) for the finetuning configuration. It uses LoRA, so it should be able to fit on 1 GPU (this example uses a GPU with 48GB memory).

Let’s first define the configuration for the two tasks:

import nemo\_run as run
from nemo.collections import llm

def configure\_checkpoint\_conversion():
    return run.Partial(
        llm.import\_ckpt,
        model\=llm.llama3\_8b.model(),
        source\="hf://meta-llama/Meta-Llama-3-8B",
        overwrite\=False,
    )

def configure\_finetuning\_recipe(nodes: int \= 1, gpus\_per\_node: int \= 1):
    recipe \= llm.llama3\_8b.finetune\_recipe(
        dir\="/checkpoints/llama3\_finetuning", \# Path to store checkpoints
        name\="llama3\_lora",
        num\_nodes\=nodes,
        num\_gpus\_per\_node\=gpus\_per\_node,
    )

    recipe.trainer.max\_steps \= 100
    recipe.trainer.num\_sanity\_val\_steps \= 0

    \# Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context\_parallel\_size \= 1
    recipe.trainer.val\_check\_interval \= 100

    \# This is currently required for LoRA/PEFT
    recipe.trainer.strategy.ddp \= "megatron"

    return recipe

You can refer to [overrides](#pretraining-overrides) for details on overriding more of the default attributes.

### Execute locally[#](#id3 "Link to this heading")

Execution should be pretty straightforward, since we will reuse the [local executor](#pretraining-local) (but include the definition here for reference). Next, we will define the experiment and launch it. Here’s what it looks like:

def local\_executor\_torchrun(nodes: int \= 1, devices: int \= 2) \-> run.LocalExecutor:
    \# Env vars for jobs are configured here
    env\_vars \= {
        "TORCH\_NCCL\_AVOID\_RECORD\_STREAMS": "1",
        "NCCL\_NVLS\_ENABLE": "0",
        "NVTE\_DP\_AMAX\_REDUCE\_INTERVAL": "0",
        "NVTE\_ASYNC\_AMAX\_REDUCTION": "1",
        "NVTE\_FUSED\_ATTN": "0",
    }

    executor \= run.LocalExecutor(ntasks\_per\_node\=devices, launcher\="torchrun", env\_vars\=env\_vars)

    return executor

def run\_finetuning():
    import\_ckpt \= configure\_checkpoint\_conversion()
    finetune \= configure\_finetuning\_recipe(nodes\=1, gpus\_per\_node\=1)

    executor \= local\_executor\_torchrun(nodes\=finetune.trainer.num\_nodes, devices\=finetune.trainer.devices)
    executor.env\_vars\["CUDA\_VISIBLE\_DEVICES"\] \= "0"

    \# Set this env var for model download from huggingface
    executor.env\_vars\["HF\_TOKEN\_PATH"\] \= "/tokens/huggingface"

    with run.Experiment("llama3-8b-peft-finetuning") as exp:
        exp.add(import\_ckpt, executor\=run.LocalExecutor(), name\="import\_from\_hf") \# We don't need torchrun for the checkpoint conversion
        exp.add(finetune, executor\=executor, name\="peft\_finetuning")
        exp.run(sequential\=True, tail\_logs\=True) \# This will run the tasks sequentially and stream the logs

\# Wrap the call in an if \_\_name\_\_ == "\_\_main\_\_": block to work with Python's multiprocessing module.
if \_\_name\_\_ \== "\_\_main\_\_":
    run\_finetuning()

The full file looks like:

import nemo\_run as run
from nemotron\_pretraining import local\_executor\_torchrun

from nemo.collections import llm

def configure\_checkpoint\_conversion():
    return run.Partial(
        llm.import\_ckpt,
        model\=llm.llama3\_8b.model(),
        source\="hf://meta-llama/Meta-Llama-3-8B",
        overwrite\=False,
    )

def configure\_finetuning\_recipe(nodes: int \= 1, gpus\_per\_node: int \= 1):
    recipe \= llm.llama3\_8b.finetune\_recipe(
        dir\="/checkpoints/llama3\_finetuning",  \# Path to store checkpoints
        name\="llama3\_lora",
        num\_nodes\=nodes,
        num\_gpus\_per\_node\=gpus\_per\_node,
    )

    recipe.trainer.max\_steps \= 100
    recipe.trainer.num\_sanity\_val\_steps \= 0

    \# Async checkpointing doesn't work with PEFT
    recipe.trainer.strategy.ckpt\_async\_save \= False

    \# Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context\_parallel\_size \= 1
    recipe.trainer.val\_check\_interval \= 100

    \# This is currently required for LoRA/PEFT
    recipe.trainer.strategy.ddp \= "megatron"

    return recipe

def local\_executor\_torchrun(nodes: int \= 1, devices: int \= 2) \-> run.LocalExecutor:
    \# Env vars for jobs are configured here
    env\_vars \= {
        "TORCH\_NCCL\_AVOID\_RECORD\_STREAMS": "1",
        "NCCL\_NVLS\_ENABLE": "0",
        "NVTE\_DP\_AMAX\_REDUCE\_INTERVAL": "0",
        "NVTE\_ASYNC\_AMAX\_REDUCTION": "1",
        "NVTE\_FUSED\_ATTN": "0",
    }

    executor \= run.LocalExecutor(ntasks\_per\_node\=devices, launcher\="torchrun", env\_vars\=env\_vars)

    return executor

def run\_finetuning():
    import\_ckpt \= configure\_checkpoint\_conversion()
    finetune \= configure\_finetuning\_recipe(nodes\=1, gpus\_per\_node\=1)

    executor \= local\_executor\_torchrun(nodes\=finetune.trainer.num\_nodes, devices\=finetune.trainer.devices)
    executor.env\_vars\["CUDA\_VISIBLE\_DEVICES"\] \= "0"

    \# Set this env var for model download from huggingface
    executor.env\_vars\["HF\_TOKEN\_PATH"\] \= "/tokens/huggingface"

    with run.Experiment("llama3-8b-peft-finetuning") as exp:
        exp.add(
            import\_ckpt, executor\=run.LocalExecutor(), name\="import\_from\_hf"
        )  \# We don't need torchrun for the checkpoint conversion
        exp.add(finetune, executor\=executor, name\="peft\_finetuning")
        exp.run(sequential\=True, tail\_logs\=True)  \# This will run the tasks sequentially and stream the logs

\# Wrap the call in an if \_\_name\_\_ == "\_\_main\_\_": block to work with Python's multiprocessing module.
if \_\_name\_\_ \== "\_\_main\_\_":
    run\_finetuning()

Here’s a recording showing all the steps above leading up to the start of finetuning:

#### Execute on a Slurm Cluster with more nodes[#](#execute-on-a-slurm-cluster-with-more-nodes "Link to this heading")

You can reuse the Slurm executor from [above](#pretraining-slurm). The experiment can then be configured like:

Note

The `import_ckpt` configuration should write to a shared filesystem accessible by all nodes in the cluster for multi-node training. You can control the default cache location by setting `NEMO_HOME` environment variable.

def run\_finetuning\_on\_slurm():
    import\_ckpt \= configure\_checkpoint\_conversion()

    \# This will make finetuning run on 2 nodes with 8 GPUs each.
    recipe \= configure\_finetuning\_recipe(gpus\_per\_node\=8, nodes\=2)
    executor \= slurm\_executor(
        ...
        nodes\=recipe.trainer.num\_nodes,
        devices\=recipe.trainer.devices,
        ...
    )
    executor.env\_vars\["NEMO\_HOME"\] \= "/path/to/a/shared/filesystem"

    \# Importing checkpoint always requires only 1 node and 1 task per node
    import\_executor \= slurm\_executor.clone()
    import\_executor.nodes \= 1
    import\_executor.ntasks\_per\_node \= 1
    \# Set this env var for model download from huggingface
    import\_executor.env\_vars\["HF\_TOKEN\_PATH"\] \= "/tokens/huggingface"

    with run.Experiment("llama3-8b-peft-finetuning-slurm") as exp:
        exp.add(importer, executor\=import\_executor, name\="import\_from\_hf")
        exp.add(recipe, executor\=executor, name\="peft\_finetuning")
        exp.run(sequential\=True, tail\_logs\=True)

On this page