# Older NVIDIA cards as peers — addendum to [README.md](README.md)

This branch is Basalt 0.4.1 with the peer-card work extended. Everything in the original README still holds; this
file covers only what is different, what it is good for, and what it does not do.

Upstream already allows one older architecture in the second slot: an **Ada (sm_89)** card as an expert-tier peer,
built with `-DPEER_ADA=ON` (PR #343). This branch takes that mechanism further.

## What is added

| | |
| --- | --- |
| `-DPEER_AMPERE=ON` | an **Ampere** card (sm_86 — RTX 30xx; sm_80 — A100, A30) may be an expert-tier peer |
| LM head on the peer | the head kernels are built for the peer architecture, which upstream's list omits |
| NVFP4 and an older peer | the FP4 tensor-core prompt share stays on the Blackwell card; the peer reads NVFP4 experts through the int8 MMQ kernels instead |

The LM head item is the one that matters in practice. The head is split by vocabulary across the two cards in the
expert-tier mechanism, not only in a layer split, whenever the peer's bandwidth earns more than a quarter of the rows
(`hs_plan()` in `src/models/qwen4exp/decode/verify/head_split.cpp`). With `card_bandwidth()` being clock x bus width,
that is any peer with **at least a third of the main card's bandwidth**. Without the head kernels built for it, such a
peer loads the whole model and then ends the run at the first head launch:

```
error: q5k_head: shared-memory opt-in failed: no kernel image is available for execution on the device
```

That gap is in the shipped sm_89 support too. An RTX 4060 Ti behind an RTX 5080 is at 0.30 of the main card's
bandwidth, just under the gate, so it never splits the head and never hit it; an RTX 4090 peer behind one Blackwell
card is at 0.56 and would.

## Hardware this supports

| Card | As the main card | As a peer (expert tier) |
| --- | --- | --- |
| RTX 50xx (sm_120) | yes | yes |
| RTX 40xx (sm_89) | no | `-DPEER_ADA=ON` |
| RTX 30xx (sm_86), A100/A30 (sm_80) | no | `-DPEER_AMPERE=ON` |
| RTX 20xx (sm_75) and older | no | no |

**The main card is still Blackwell.** Nothing here changes that. Five files on the main path use sm_90+ features with
no fallback — thread-block clusters in `src/kernels/cuda/dense_ops.cu`, raw mbarrier PTX in
`src/models/qwen4exp/kernels/hc_fused.cu`, `cp.async.bulk.prefetch` in `qwen4exp/kernels/verify_kernels.cu`, and TMA
pipelines in `src/prefill/fp8_gemm.cu` and `src/prefill/fp4_dense.cu` — so an older main card needs real alternative
implementations, not a build flag.

**Turing (RTX 20xx) is not a peer either.** The peer kernels use `cp.async`, the `m16n8k32` int8 MMA shape and bf16,
all of which start at sm_80.

## Building

Both options are off by default, so a Blackwell-only build is unchanged and no slower than upstream's.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPEER_ADA=ON -DPEER_AMPERE=ON
cmake --build build -j
```

Each option adds its architecture to the 36 peer translation units, so the full build above compiles those four times
over. On 24 cores that is hours rather than the ten minutes a Blackwell-only build takes. Enable only the
architecture you own, unless you are building a binary for other people.

Verify a build carries what you think it does — the build log is not evidence, the objects are:

```sh
cuobjdump --list-elf bin/basalt | grep -c sm_86
```

## Running: choosing the cards

No new command-line argument. The peer work changes which cards the existing `gpu` setting will **accept**, not
how you write it.

```sh
basalt serve --gpu 0,2       # exactly these, main card first, the rest expert tiers
basalt serve --gpu auto      # every card this build can use; main card by bandwidth, then VRAM
basalt serve                 # unset: card 0 alone, with a warning naming the cards left out
basalt --gpu 0,2 --gpu-plan  # print the plan as JSON and exit. Bare `basalt` only: under `serve` the
                             # flag is accepted, ignored, and the whole model loads
```

The same string goes in the config instead, per model:

```json
{ "models": [{ "model": "models/your-pack.basalt", "gpu": "0,2", "kv": "int8" }] }
```

**The numbers are `nvidia-smi`'s**, not CUDA's. Order matters: the first is the main card and must be Blackwell;
every later one is an expert tier. `--gpu 2,0` with an Ampere card first is refused, by design.

`auto` takes every card the binary has kernels for, so what it picks depends on how you built it: a default build
skips the older cards with one line each, a `-DPEER_AMPERE=ON` build takes them as tiers. Name the cards
explicitly if some other program needs one of them — `auto` will claim it.

Check the start, which says what each card became:

```
gpus: RTX 5060 Ti (nvidia-smi 0) main, RTX 3060 (nvidia-smi 2) expert tier
multi-gpu: 2 cards unlike; ms per window: expert tier 28.3, layer split 34.6, tensor parallel 25.7 (not built); picked expert tier
peer GPU 1: 6354 experts, 10.28 GiB (read in the start's pass)
verify: the head is split by vocabulary: CUDA0 148480 rows, the peer 99840 rows
generate: experts live in VRAM 7.8 GiB (19%), peer GPU 10.3 GiB (26%), RAM 21.9 GiB (55%), file (SSD) 0.0 GiB (0%)
```

`peer GPU 1` is the card's position in the run, not its `nvidia-smi` number. A card this build cannot use is
refused at the start, naming the flag to rebuild with, instead of dying at its first kernel:

```
gpu: GPU 1 (RTX 3090) is sm_86 (rebuild with -DPEER_AMPERE=ON to enable sm_86 peer support)
```

The knobs that matter once a peer is in play, none of them new:

| Flag | What it does |
| --- | --- |
| `--peer-reserve-mib N` | VRAM the tier leaves free on its own card (default 600). Raise it if the start says `prefill peer: the peer's buffers do not fit`, or if that card also drives a display |
| `--peer-exclusive` | experts on the peer are held *only* there, with no RAM copy. Buys tier size, costs file reads when the prompt path needs them. `auto` turns it on when RAM is short |
| `--no-peer-prefill` | the peer computes no prompt rows; the prompt stays on the main card |
| `--resident-experts` | keep the RAM tier page-locked rather than reading experts from the pack |

One thing to remember across card changes: `models/expert-profile.bin` records where each expert sat. Adding,
removing or swapping a card invalidates it. Copy it aside before re-tuning — a tuning run rewrites it in place.

## What an older peer does and does not do

Does: holds experts and computes their rows every decode window; reads its experts' share of each prompt chunk (up to
2,559 rows when it is the slower card); takes its share of the LM head and of the GDN mixers when its bandwidth earns
one.

Does not: run the dense weights, the KV cache or the drafter — those are the main card's; take the FP4 tensor-core
prompt share of an NVFP4 pack (int8 MMQ instead: correct, not FP4 speed); serve DeepSeek, whose peer-tier kernels are
Blackwell-only and which refuses an older peer at the start; take part in a layer split, overlapped windows or tensor
parallel, all of which need matched Blackwell cards.

Note that the expert profile (`models/expert-profile.bin`) records where each expert sat. Adding or removing a card
invalidates it — keep a copy per card layout and re-tune after a change.

## Measured

RTX 5060 Ti 16 GB (main, 180 W) + RTX 3060 12 GB (peer, 170 W), Windows, CUDA 13.0, Qwen3.8-Flash-Next IQ3_XXS pack,
PCIe gen3 x4 to the peer:

| | 5060 Ti alone | with the 3060 peer |
| --- | --- | --- |
| decode, sustained | 42–53 tok/s | 65–80 tok/s |
| prefill, cold | — | 1,129–1,381 tok/s |
| experts on the SSD | 22.1 GiB | 0 |
| full load | 60–80 s | 122 s (see below) |

The head really is shared: `verify: the head is split by vocabulary: CUDA0 148480 rows, the peer 99840 rows`,
40.2% of the vocabulary on the peer.

The load is slower than it looks: with no SSD tier the start reads 25.35 GB from the pack and page-locks
23.61 GiB of it, which is most of those 122 s. The earlier 42 s load left 22 GiB of experts on the SSD and
never read them, paying for it later on every miss instead.

Under generation the 3060 ran 86% busy against the 5060 Ti's 39% at 9% memory activity: the peer is compute-bound and
sets the pace, since the head split hands it rows by bandwidth while it has about half the shader throughput. The x4
link moved 0.4 GB/s and was never a constraint.

## Not yet verified

- sm_80 (A100, A30): compiled, never run — no such card here.
- The NVFP4 fallback through MMQ: reasoned from the code and compiled, not run against an NVFP4 pack.
- Three or more cards with an older peer among them.

## Where this fork is going

The aim is a Basalt that uses whatever NVIDIA cards you own. One half of that is done and is what this branch
contains: **any supported card as a peer**, holding and computing the expert tier beside a Blackwell main card.

The other half is **a non-Blackwell card as the main card**, and it is not done. It is not a build flag away
either. Five files on the main path use sm_90+ features with no fallback:

| File | Feature |
| --- | --- |
| `src/kernels/cuda/dense_ops.cu` | `__cluster_dims__`, cooperative-groups clusters |
| `src/models/qwen4exp/kernels/hc_fused.cu` | raw mbarrier PTX, the hyper-connection read |
| `src/models/qwen4exp/kernels/verify_kernels.cu` | `cp.async.bulk.prefetch` |
| `src/prefill/fp8_gemm.cu` | TMA tiles through an mbarrier ring |
| `src/prefill/fp4_dense.cu` | TMA + mbarrier pipeline |

Each needs an alternative implementation, and an older main card gives up NVFP4 (Blackwell-only FP4 tensor
cores), FP8 on Ampere, and the PDL launch overlap that compiles to nothing below sm_90. Most of the rest of
the engine already branches on `__CUDA_ARCH__` and would follow.

Until that exists, the honest claim for this fork is the one above: mixed-GPU inference with a Blackwell main
card, and every other card put to work.

## Known wrinkle

The cards are named by taking the `CUDA_VISIBLE_DEVICES` token as an NVML index (`src/program/gpu_select.cpp`).
Both lists are in bus order, so they agree — unless NVML enumerates a card CUDA will not give you, such as one
that has fallen off the bus (`GPU is lost`). Every index after it is then off by one: the start line cannot name
the peer, and the dashboard asks for the missing card and draws one panel. A reboot restores both.
