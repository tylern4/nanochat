"""ClearML pipeline stage wrapper for nanochat.

Runs one nanochat pipeline stage as a ClearML task, handing state between
stages via ClearML artifacts (tokenizer dirs, model checkpoints) that are
stored on the task's output storage (e.g. s3://dasml/clearml via output_uri).

Usage (from the repo root):
    python -m scripts.clearml_stage --stage tok
    python -m scripts.clearml_stage --stage base --upstream-tok <task_id> \
        --depth 20 --num-iterations 3800 --device-batch-size 8
    python -m scripts.clearml_stage --stage sft --upstream-base <task_id>
    python -m scripts.clearml_stage --stage chat_eval --upstream-sft <task_id>

Register mode (used by the pipeline controller to create the task without
running anything): append --register.

Env:
    CLEARML_PROJECT   project name (default "nanochat")
    NPROC             torchrun processes (default 4)
    NANOCHAT_BASE_DIR working dir for shards/checkpoints (default /tmp/nanochat)
    WF_OUTPUT_URI     task output_uri (artifacts/checkpoints destination)
"""

import argparse
import os
import shutil
import subprocess
import sys

# NOTE: Task.init must happen before anything heavy; the module-level flow below
# is deliberately linear so the pipeline controller can register/attach cleanly.

STAGES = ("tok", "base", "sft", "chat_eval")


def sh(cmd: list, env: dict) -> None:
    """Run a subprocess, streaming output; raise on failure."""
    print0 = lambda m: print(m, flush=True)
    print0(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="nanochat ClearML stage")
    # NOTE: all args optional - when run remotely the agent re-executes this
    # script with no argv; the actual values come from the connected params.
    parser.add_argument("--stage", choices=STAGES, default=None)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--upstream-tok", type=str, default="")
    parser.add_argument("--upstream-base", type=str, default="")
    parser.add_argument("--upstream-sft", type=str, default="")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=48, help="pretraining shards to download")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args, _ = parser.parse_known_args()

    from clearml import Task

    project = os.environ.get("CLEARML_PROJECT", "nanochat")
    task = Task.init(
        project_name=project,
        task_name=f"nanochat-{args.stage}",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks=False,
        reuse_last_task_id=False,
    )
    params = task.connect(
        {
            "stage": args.stage,
            "depth": args.depth,
            "num_iterations": args.num_iterations,
            "device_batch_size": args.device_batch_size,
            "num_shards": args.num_shards,
            "max_seq_len": args.max_seq_len,
            "upstream_tok": args.upstream_tok,
            "upstream_base": args.upstream_base,
            "upstream_sft": args.upstream_sft,
            "nproc": int(os.environ.get("NPROC", "4")),
        },
        name="stage",
    )

    output_uri = os.environ.get("WF_OUTPUT_URI")
    if output_uri:
        task.output_uri = output_uri

    if args.register:
        print(f"register stage task: id={task.id}")
        task.close()
        return

    task.execute_remotely(queue_name=os.environ.get("CLEARML_QUEUE", "muller"))

    stage = params["stage"]
    if stage not in STAGES:
        raise RuntimeError(f"missing/invalid stage parameter: {stage!r}")

    # ------------------------------------------------------------------
    # running remotely (or locally after the handoff): do the work
    # ------------------------------------------------------------------
    base_dir = os.environ.get("NANOCHAT_BASE_DIR") or (
        # prefer real scratch over /tmp (tmpfs/RAM on the node). SCRATCH is
        # set on NERSC batch hosts but may not propagate into the container;
        # fall back to /tmp so the wrapper stays portable across machines.
        os.path.join(os.environ["SCRATCH"], "nanochat")
        if os.environ.get("SCRATCH") else "/tmp/nanochat"
    )
    os.environ["NANOCHAT_BASE_DIR"] = base_dir  # inner scripts must agree
    os.makedirs(base_dir, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["CLEARML_RUN"] = "1"  # inner nanochat scripts log via ClearML shim

    nproc = int(os.environ.get("NPROC", "4"))
    import torch

    if torch.cuda.is_available():
        nproc = min(nproc, torch.cuda.device_count())

    def download_artifact(upstream_id: str, name: str, dest: str) -> None:
        """Fetch a folder artifact from an upstream task and place it at dest."""
        upstream = Task.get_task(task_id=upstream_id)
        art = upstream.artifacts.get(name)
        if art is None:
            raise RuntimeError(f"upstream task {upstream_id} has no artifact '{name}'")
        local = art.get()  # downloads + extracts folder artifacts
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(local, dest)
        print(f"restored {name} from {upstream_id} -> {dest}")

    def torchrun(module: str, extra: list) -> list:
        return [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={nproc}", "-m", module, "--",
        ] + extra

    if stage == "tok":
        # tokenizer data (~8 shards, ~800MB) then train + evaluate
        sh([sys.executable, "-m", "nanochat.dataset", "-n", "8"], os.environ)
        sh([sys.executable, "-m", "scripts.tok_train"], os.environ)
        sh([sys.executable, "-m", "scripts.tok_eval"], os.environ)
        from pathlib import Path
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        task.upload_artifact("tokenizer", Path(tokenizer_dir), wait_on_upload=True)

    elif stage == "base":
        if params["upstream_tok"]:
            download_artifact(params["upstream_tok"], "tokenizer", os.path.join(base_dir, "tokenizer"))
        # pretraining shards (each ~250M chars); downloaded directly from HF
        sh([sys.executable, "-m", "nanochat.dataset", "-n", str(params["num_shards"])], os.environ)
        extra = [
            f"--depth={params['depth']}",
            f"--max-seq-len={params['max_seq_len']}",
            f"--device-batch-size={params['device_batch_size']}",
        ]
        if params["num_iterations"] and params["num_iterations"] > 0:
            extra.append(f"--num-iterations={params['num_iterations']}")
        sh(torchrun("scripts.base_train", extra), os.environ)
        from pathlib import Path
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", f"d{params['depth']}")
        task.upload_artifact("base-checkpoint", Path(checkpoint_dir), wait_on_upload=True)

    elif stage == "sft":
        if params["upstream_tok"]:
            download_artifact(params["upstream_tok"], "tokenizer", os.path.join(base_dir, "tokenizer"))
        if params["upstream_base"]:
            download_artifact(params["upstream_base"], "base-checkpoint",
                              os.path.join(base_dir, "base_checkpoints", f"d{params['depth']}"))
        # chat_sft auto-guesses the base checkpoint tag/step and inherits hyperparams
        sh(torchrun("scripts.chat_sft", []), os.environ)
        from pathlib import Path
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", f"d{params['depth']}")
        task.upload_artifact("sft-checkpoint", Path(checkpoint_dir), wait_on_upload=True)

    elif stage == "chat_eval":
        if params["upstream_tok"]:
            download_artifact(params["upstream_tok"], "tokenizer", os.path.join(base_dir, "tokenizer"))
        if params["upstream_sft"]:
            download_artifact(params["upstream_sft"], "sft-checkpoint",
                              os.path.join(base_dir, "chatsft_checkpoints", f"d{params['depth']}"))
        sh(torchrun("scripts.chat_eval", ["-i", "sft"]), os.environ)

        # generate a few chat samples: Plots table + debug text samples
        import io

        import torch
        from nanochat.checkpoint_manager import load_model
        from nanochat.engine import Engine

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer, meta = load_model("sft", device, phase="eval")
        engine = Engine(model, tokenizer)
        bos = tokenizer.get_bos_token_id()
        user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
        assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

        prompts = [
            "Why is the sky blue?",
            "What is the capital of France?",
            "If 5*x + 3 = 13, then x is",
            "Tell me a one-paragraph story about a robot.",
        ]
        rows = [["prompt", "response"]]
        logger = task.get_logger()
        for i, prompt in enumerate(prompts):
            conversation = [bos, user_start]
            conversation.extend(tokenizer.encode(prompt))
            conversation.extend([user_end, assistant_start])
            response_tokens = []
            for token_column, _masks in engine.generate(
                conversation, num_samples=1, max_tokens=256, temperature=0.6, top_k=50
            ):
                response_tokens.append(token_column[0])
            if response_tokens[-1] != assistant_end:
                response_tokens.append(assistant_end)
            response = tokenizer.decode(response_tokens)
            rows.append([prompt, response])
            logger.report_media(
                title="generations", series=prompt[:40], iteration=i,
                stream=io.BytesIO(f"prompt: {prompt}\n\nresponse: {response}".encode()),
                file_extension="txt",
            )
        logger.report_table("chat_samples", "sft", iteration=0, table=rows)
        logger.report_text(f"chat samples generated: {len(prompts)}")

    print(f"stage '{stage}' complete")


if __name__ == "__main__":
    main()
