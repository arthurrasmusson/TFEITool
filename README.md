# TFEITool: TensorRT‑LLM KV Cache Inspector & Slide Generator

`TFEITool.py` is a standalone Python program that turns a real or simulated inference
request into a self‑contained PowerPoint presentation.  The slides
explain how a text prompt flows through a TensorRT‑LLM inference server,
visualise the resulting key/value (KV) cache pages and binary files, and
include timing diagrams for the underlying GPU kernels and I/O.

The tool generated the “Future of Exabyte‑Scale Inference” presentation
and aims to make complex memory layouts and paging strategies easy to
understand.  It implements the full stack shown in the slide builder
block diagrams: from an OpenAI‑compatible HTTP client, through the
TensorRT‑LLM runtime, down to the GPUDirect Storage offload path, and
back up as a labelled hex dump.  When run in `--dry-run` mode the
tool synthesises data so it can run anywhere without a live server.

## Features

* **Inference & I/O tracing** – Connects to a TensorRT‑LLM 1.0+ server via
  its OpenAI‑compatible `/v1/chat/completions` API or performs a
  synthetic run.  It records time to first token, token rate and
  upload/download bytes.
* **KV geometry discovery** – Reads the engine configuration to obtain
  the number of KV heads (`Hk`), tokens per block (`T`), head
  dimension (`D`) and dtype.  Alternatively these can be specified on
  the command line.
* **Cache binary inspection** – Loads a `block_pool.bin` file and maps
  each byte back to `(kv, head, token, dim)` using the row‑major
  layout `[2,K,V], Hk, T, D` and the formula:

  ```
  idx = (((kv * Hk + h) * T) + t) * D + d
  off = idx * dtype_bytes
  ```

* **Timeline visualisation** – If provided with an `nsys` trace (`.qdrep`)
  or NVTX JSON, the tool parses NVTX ranges and produces Gantt
  diagrams showing Prefill/Decode stages, attention kernels,
  Onboard/Offload/copyBlock events and I/O (GDS, cuFileRead/Write).
* **Clean slide layout** – Generates a 16×9 PPTX with no overlaps or
  clipped text.  Slides include: legend & geometry, tile maps for K/V
  planes, linear on‑disk layout, zoom‑in of a 32‑byte block, full
  non‑truncated hex pages with colour bars for plane/head and token
  stripes, timeline diagrams, and a reproducibility summary.

## Installation

This repository contains two files:

- `TFEITool.py`: the Python program described above.
- `README.md`: this documentation.

To run the tool you need Python 3.8 or newer with the following
packages:

```
pip install python-pptx matplotlib numpy
```

If you want to run against a real TensorRT‑LLM engine you must also
have the `tensorrt‑llm` Python package and an OpenAI‑compatible
inference server running (e.g. `trtllm-serve`).  To generate
timelines, install NVIDIA Nsight Systems and capture a trace with
`nsys profile -t cuda,nvtx`.

## Usage

### Synthetic dry run

```
python TFEITool.py --dry-run \
  --prompt "example audience prompt" \
  --output tfei_report.pptx
```

This will produce `tfei_report.pptx` with synthetic KV data and
timeline slides.

### Real inference with engine discovery

```
python TFEITool.py \
  --server http://localhost:8000 \
  --engine-dir /path/to/engine \
  --kv-binary /path/to/block_pool.bin \
  --prompt "illustrating inference data flow" \
  --output tfei_report.pptx
```

Replace `/path/to/engine` and `/path/to/block_pool.bin` with your
actual engine directory and KV cache dump.  If you captured a trace
with Nsight Systems, add `--qdrep session.qdrep` to overlay the
timeline diagrams.

### Command‑line options

- `--prompt` / `--prompt-file` – user prompt to run through the
  inference server.
- `--kv-dims` – override the KV layout (default `2,8,32,16`).
- `--dtype-bytes` – bytes per element (2 for fp16, 1 for fp8/int8, 4 for fp32).
- `--dry-run` – skip server calls and generate synthetic data.
- `--output` – path to the generated PPTX file.
- `--qdrep` / `--timeline-json` – import an NSYS `.qdrep` or NVTX
  JSON trace for timeline slides.
- `--kv-binary` – explicit path to a KV block binary to analyse.
- `--strict-trtllm` – fail if the TRT‑LLM Python bindings are not
  installed.

