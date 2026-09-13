# Sources and attribution

- FlyType is a fork of [nftechie/stonkfly](https://github.com/nftechie/stonkfly),
  copyright © 2026 nftechie and DOOMFLY contributors, under the MIT License
  preserved in `LICENSE`. The fork is pinned to upstream commit
  `78ef3e05ab0fa086032098558d893667068944a0`. FlyType removes all trading,
  Coinbase, AgentKit and account/ledger code and adds a character-selection
  task, fixture-mode testing, and a LEFT/RIGHT/HOLD neural decoder in its
  place; the retained MaleCNS v1.0 importer, inferred visual projection,
  spiking kernel and baseline-centered plasticity implementation are otherwise
  unchanged from Stonkfly, which itself adapted them from
  [DOOMFLY](https://github.com/nftechie/doomfly) (also MIT). No Doom game
  assets, World Labs assets, movie clips, or portfolio material are included.
- [MaleCNS v1.0](https://male-cns.janelia.org/) data: the MaleCNS collaboration
  and upstream contributors, distributed under the release's Creative Commons
  Attribution 4.0 terms. Downloaded separately; see
  `flytype/neural/datasets.json` and `flytype/neural/sources.lock.json` for
  release URLs and checksums. Cite the dataset and its associated paper when
  publishing results. FlyType's retained graph and modeled physiology are
  derived interpretations, not an official dataset product.
- Scientific sources informing the model are linked in `docs/model.md`. No
  papers or figures are redistributed.
- `assets/stonkfly.png` is upstream Stonkfly project artwork retained
  unchanged; its generation prompt is in `assets/image-generation.md`.
