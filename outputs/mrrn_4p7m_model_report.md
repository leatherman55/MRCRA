# MRRN 4.7M production language configuration

## Result

The new production FineWeb model contains **4,695,023 trainable parameters**
with the 50,257-token GPT-2 vocabulary. This is 82.24% smaller than the legacy
26,439,515-parameter configuration and is within 0.11% of the requested 4.7M
target.

## Configuration

| Component | Production value |
|---|---:|
| token/model width | 48 |
| blocks | 3 |
| resolutions | 5 |
| scale widths | 48, 64, 64, 64, 64 |
| attention heads | 4 |
| base recurrent modes | 10 |
| scale modes | 10, 12, 12, 12, 12 |
| complex MIMO rank | 2 |
| RSGLU spectral modes | 8 |
| RSGLU basis order | 4 |
| legal triads per mode | 1 |
| mixer expansion | 2.0 |
| local attention window | 16 |
| retrieved items | 8 |
| eidetic capacity | 2,048 |
| global prediction head | disabled |

The model retains tied input/output embeddings, exact learned lifting, five
physical resolutions, paired-real complex resonators, local and coherence
attention, coarse landmarks, bounded eidetic retrieval, and learned spectral
activation triads. Only capacity was reduced; the architectural mechanisms were
not replaced by simpler surrogates.

## Parameter allocation

| Allocation | Parameters | Share |
|---|---:|---:|
| tied embedding and output bias | 2,462,593 | 52.45% |
| three multiresolution blocks | 2,159,154 | 45.99% |
| lifting, adapters, raw mixer, memory projections, norms, gates | 73,276 | 1.56% |
| **total** | **4,695,023** | **100%** |

Width 48 is intentional. The GPT-2 embedding alone scales as `50,257 × width`;
shrinking width much further would save parameters efficiently but would also
make each of four heads too narrow. At width 48 each head retains 12 channels.
The deeper scales grow once to width 64 and retain 12 recurrent modes, while
eight RSGLU modes preserve most of the spectral nonlinear capacity of the old
model. Three blocks were preferred over aggressively narrowing a five-block
network because token representation and head width are global bottlenecks.

## Training implications

- 8M tokens provide 1.70 tokens per parameter.
- 20M tokens provide 4.26 tokens per parameter.
- FP32 weights occupy about 17.9 MiB.
- FP32 weights, gradients, and two Adam moments total about 71.6 MiB before
  allocator, fused-optimizer, activation, and telemetry overhead.
- At a fixed sequence/batch shape, activation memory and runtime should fall
  substantially, but no RTX A4500 throughput number is claimed until measured.
- Recurrent generation remains constant-state with respect to generated context;
  the 32,768-token interface does not allocate a Transformer-style KV cache.

## Compatibility boundary

New runs use `fineweb_4p7m_config`, output directory
`outputs/fineweb-4p7m-stable`, and Trackio names beginning `mrrn-4p7m-`.
The legacy `fineweb_27m_config` is retained only for loading or evaluating old
26.4M checkpoints. Parameter shapes differ, so a 26.4M optimizer/model checkpoint
cannot be resumed into the 4.7M model. Starting in a separate directory prevents
accidental metric and checkpoint mixing.
