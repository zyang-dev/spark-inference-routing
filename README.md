# Route tasks with Switchyard on DGX Spark

The idea is to get more work done with the same DGX Spark hardware, without sacrificing answer quality. Switchyard sends tasks that MiniCPM5-2B can handle to the smaller model and uses Qwen3.8-Flash-Next-NVFP4 for more demanding work. We then compare the results with running everything on Qwen.

This example uses one Qwen replica across two DGX Sparks and one MiniCPM replica on a third Spark. The same setup can scale to more nodes: add model replicas and use a load balancer to distribute requests within each model pool.

This tutorial builds [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard), loads a routing policy, and runs tool-eval-bench against two model backends:

- **Qwen3.8-Flash-Next-NVFP4:** one replica running across two DGX Sparks.
- **MiniCPM5-2B:** one replica running on one DGX Spark.

MiniCPM also classifies requests to choose the executor. Classification has thinking disabled; MiniCPM task execution has thinking enabled. Both roles share the same loaded MiniCPM model through two API aliases.

```text
 tool-eval-bench
       |
       v
 Switchyard :4000 (Spark 3)
       |
       +-- MiniCPM judge: choose the executor (thinking off) --> Spark 3
       |
       +-- MiniCPM executor (thinking on) --> Spark 3
       |
       +-- Qwen executor -----------------> Sparks 1 and 2
```

First start Qwen on Sparks 1 and 2, then MiniCPM on Spark 3. Next, build and run Switchyard on Spark 3 alongside MiniCPM. Run the benchmark on Spark 3 as well. Spark 3 must be able to reach the Qwen API on Spark 1.

## 1. Start Qwen on two DGX Sparks

Use [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) and its [Qwen3.8 Flash Next NVFP4 cluster recipe](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-flash-next-nvfp4-cluster.yaml).

Spark 1 is the head node and exposes the API. Spark 2 is the worker. Together they run one Qwen replica; Spark 3 is reserved for MiniCPM.

### Prepare the two-Spark cluster

For network setup, we recommend using the **Cluster Assistant** in [NVIDIA Sync](https://docs.nvidia.com/sync/latest/index.html). Connect Sparks 1 and 2 over their high-speed interfaces and ensure passwordless SSH is configured. The repository's [Networking Guide](https://github.com/eugr/spark-vllm-docker/blob/main/docs/NETWORKING.md) provides additional setup and troubleshooting instructions. Both nodes need working GPU-enabled Docker access as described in the repository.

On **Spark 1**, clone the repository into a separate directory from Switchyard:

```bash
cd ~
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

If it is already cloned, enter that existing checkout instead.

Discover the cluster and save its configuration:

```bash
./run-recipe.sh --discover
```

Confirm that the discovered cluster contains only Sparks 1 and 2, with Spark 1 first. Do not include the MiniCPM node. An existing two-node configuration can be reused.

### Launch the recipe

From the `spark-vllm-docker` directory on Spark 1:

```bash
./run-recipe.sh recipes/qwen3.8-flash-next-nvfp4-cluster.yaml --setup
```

The setup prepares the required image and model when missing, distributes them to the cluster, and launches the server. Allow time for model download, loading, and warmup.

This tutorial uses the current serving recipes and latest images: `eugr/spark-vllm-b12x:latest` for Qwen and `eugr/spark-vllm:latest` for MiniCPM. These may differ from the serving environment used for the historical benchmark results below.

### Check the Qwen API

After the server is ready, run this from Spark 3. Replace `QWEN_API_HOST` with Spark 1's reachable API address:

```bash
curl -fsS http://QWEN_API_HOST:8000/v1/models
```

Confirm that the response lists `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`. Keep Spark 1's API address for the Switchyard policy in step 4. The model server coordinates execution across both Qwen Sparks.

## 2. Start MiniCPM on one DGX Spark

Use the same [spark-vllm-docker repository](https://github.com/eugr/spark-vllm-docker), with a custom MiniCPM recipe. Run these commands on **Spark 3**:

```bash
cd ~
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

If it is already cloned on Spark 3, enter the existing checkout instead.

### Download the MiniCPM recipe

Download [minicpm5-2b-spark.yaml](https://github.com/zyang-dev/spark-inference-routing/blob/main/spark-vllm-docker-recipes/minicpm5-2b-spark.yaml) from this repository and save it as `recipes/minicpm5-2b-spark.yaml` in your `spark-vllm-docker` checkout on Spark 3.

### Start the server

```bash
./run-recipe.sh recipes/minicpm5-2b-spark.yaml --solo --setup
```

Wait for the model to load and the API server to become ready. The selected image must support the `minicpm5` tool parser.

Thinking is disabled by default for requests that omit an override. The policy explicitly sends:

- `enable_thinking=false` for the `minicpm5-judge` alias.
- `enable_thinking=true` for the `openbmb/MiniCPM5-2B` executor alias.

Both aliases use the same loaded weights. The reasoning parser separates reasoning from the final answer, which matters for tests requiring valid JSON or an exact output format.

### Check the API

In another terminal on Spark 3, check the endpoint:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
```

Confirm that both `openbmb/MiniCPM5-2B` and `minicpm5-judge` are listed. A missing judge alias causes an upstream “model does not exist” error. Use this endpoint in the policy's `minicpm_vllm` client.

## 3. Build Switchyard on Spark 3

Run this section on **Spark 3**, alongside the MiniCPM server.

These steps assume the standard DGX Spark environment has the system build tools and Python installed.

Check whether Rust and Cargo are available:

```bash
rustc --version
cargo --version
```

If either command reports `command not found`, install Rust and Cargo with [rustup](https://rustup.rs/), then load its shell environment:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/switchyard-rustup-init.sh
sh /tmp/switchyard-rustup-init.sh -y
. "$HOME/.cargo/env"
```

Clone [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) and build its server:

```bash
cd ~
git clone https://github.com/NVIDIA-NeMo/Switchyard.git
cd Switchyard
git checkout --detach 3cf60ce07e52c159d9c133ae78cd8c40fbc12ef3
cargo build --release --locked -p switchyard-server
```

Run subsequent Switchyard commands from this checkout. Rustup uses the toolchain specified by the repository's `rust-toolchain.toml`. `--locked` requires Cargo to keep the dependency versions recorded in `Cargo.lock`.

Check that the server binary runs:

```bash
./target/release/switchyard-server --help
```

The build above pins Switchyard to commit [`3cf60ce`](https://github.com/NVIDIA-NeMo/Switchyard/commit/3cf60ce07e52c159d9c133ae78cd8c40fbc12ef3), the source revision used for this tutorial.

## 4. Download and configure the policy

From `~/Switchyard` on **Spark 3**, use **`mincpm-qwen-spark.toml`**. The MiniCPM judge and executor share one vLLM instance; Qwen runs separately.

Download the [Spark routing policy](https://github.com/zyang-dev/spark-inference-routing/blob/main/switchyard-policies/mincpm-qwen-spark.toml):

```bash
curl -fL \
  'https://raw.githubusercontent.com/zyang-dev/spark-inference-routing/main/switchyard-policies/mincpm-qwen-spark.toml' \
  -o mincpm-qwen-spark.toml
```

Edit the two backend addresses in that file:

```toml
[llm_clients.minicpm_vllm]
format = "openai_chat"
base_url = "http://127.0.0.1:8000/v1"

[llm_clients.qwen_vllm]
format = "openai_chat"
base_url = "http://QWEN_API_HOST:8000/v1"
```

Replace `QWEN_API_HOST` with the API-serving host of the two-Spark Qwen replica. The two Sparks form one model replica with one API endpoint.

MiniCPM and Switchyard both run on Spark 3, so the MiniCPM URL uses `127.0.0.1`. Keep `/v1` in both backend URLs. Update the Qwen address in the downloaded policy to match Spark 1.

The policy provides these client-facing model names:

| Model name | Behavior |
| --- | --- |
| `switchyard/agent` | Classify the request, then route to MiniCPM or Qwen |
| `switchyard/minicpm` | Call MiniCPM directly without classification |
| `switchyard/qwen` | Call Qwen directly without classification |

The judge and MiniCPM executor settings must remain distinct:

```toml
[targets.minicpm]
id = "openbmb/MiniCPM5-2B"
llm_client = "minicpm_vllm"
extra_body = { chat_template_kwargs = { enable_thinking = true } }

[targets.minicpm_judge]
id = "minicpm5-judge"
llm_client = "minicpm_vllm"
extra_body = { chat_template_kwargs = { enable_thinking = false } }
```

These settings are already in the supplied policy; do not append duplicate sections. Keep both aliases so the judge and executor can use different thinking settings.

## 5. Start Switchyard and run tool-eval-bench

### Validate and start Switchyard

On **Spark 3**, open a terminal in `~/Switchyard` and validate the policy:

```bash
./target/release/switchyard-server \
  --config ./mincpm-qwen-spark.toml \
  --dry-run
```

This validates configuration; it does not replace the backend connectivity checks above.

Start the router and keep this terminal running:

```bash
set -o pipefail
RUST_LOG=info ./target/release/switchyard-server \
  --config ./mincpm-qwen-spark.toml \
  --host 0.0.0.0 \
  --port 4000 \
  2>&1 | tee switchyard-server.log
```

The command listens on all host interfaces so another machine can run the benchmark. Use `--host 127.0.0.1` if all clients run locally. This example assumes a trusted internal network.

### Check a routed request

In a second terminal on Spark 3, run `cd ~/Switchyard`, then send a test request. The examples use localhost because the benchmark runs on the same host:

```bash
curl -fsS http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-switchyard-session-id: tutorial-smoke-test' \
  -d '{
    "model": "switchyard/agent",
    "messages": [{"role": "user", "content": "Say hello briefly."}]
  }'
```

Check `switchyard-server.log` for the judge call and selected executor. A valid answer confirms that the selected backend works; test the direct model routes as well to check both backends.

### Install the benchmark

Install [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) in an isolated Python environment:

```bash
python3 -m venv .venv-tool-eval
. .venv-tool-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install 'git+https://github.com/SeraphimSerapis/tool-eval-bench.git@3950aab24'
python -m pip freeze > tool-eval-environment.txt
tool-eval-bench run --help
```

### Run a small check

```bash
tool-eval-bench run \
  --base-url http://127.0.0.1:4000 \
  --model switchyard/agent \
  --session-header x-switchyard-session-id \
  --scenarios TC-01 TC-47 \
  --seed 42 \
  --timeout 120
```

Keep the session-header option for multi-turn tests. It lets the benchmark identify each conversation to Switchyard. The policy classifies on user turns and reuses decisions on intervening tool steps.

### Run the full routed benchmark, including hard cases

```bash
set -o pipefail
tool-eval-bench run \
  --base-url http://127.0.0.1:4000 \
  --model switchyard/agent \
  --session-header x-switchyard-session-id \
  --hardmode \
  --seed 42 \
  --timeout 120 \
  2>&1 | tee benchmark-agent.log
```

Omit `--scenarios` to run the entire suite. For a benchmark client on another host, replace `127.0.0.1` with the Switchyard host's address.

### Run the Qwen-only baseline

Use the same benchmark version, seed, and serving settings:

```bash
tool-eval-bench run \
  --base-url http://127.0.0.1:4000 \
  --model switchyard/qwen \
  --session-header x-switchyard-session-id \
  --hardmode \
  --seed 42 \
  --timeout 120 \
  2>&1 | tee benchmark-qwen.log
```

For a MiniCPM-only comparison, use `--model switchyard/minicpm`.

Keep the benchmark's generated reports, terminal output, Switchyard logs, policy, source revisions, and model-serving commands together. Compare scores and failures as well as timing. A sequential quality benchmark does not measure maximum concurrent throughput.

## Reference result from the original experiment

Switchyard source revision: [`3cf60ce07e52c159d9c133ae78cd8c40fbc12ef3`](https://github.com/NVIDIA-NeMo/Switchyard/commit/3cf60ce07e52c159d9c133ae78cd8c40fbc12ef3).

These results were obtained with **tool-eval-bench `v2.7.1.dev20+g3950aab24`**, which included **92 scenarios with Hard Mode enabled**. Newer benchmark versions may contain additional tests or grading changes, so these scores apply to that specific version. They are single-run results, not guaranteed outcomes for a new installation:

| Metric | Qwen-only | Switchyard agent |
| --- | ---: | ---: |
| Scenarios, including hard cases | 92 | 92 |
| Points | 168/184 | 168/184 |
| Passed | 80 | 79 |
| Partial | 8 | 10 |
| Failed | 4 | 3 |

The agent matched Qwen's aggregate points while sending some execution calls to MiniCPM. Matching total points does not mean both runs passed the same scenarios.
