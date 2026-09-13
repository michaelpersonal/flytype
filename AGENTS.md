# FlyType

- Preserve the full retained MaleCNS v1.0 graph. No pruning, scripted characters presented as neural output, LLM selection policy, or hidden target-text override of the decoded action.
- Separate target-text state, sensory rendering, neural propagation, plasticity, and fixed decoding. `task.py`/`feedback.py` never enter `neural/`; `neural/controller.py` never reads target text or correctness.
- Reward/aversive reinforcement is an engineered input to identified dopamine cells. Do not claim modeled pain, pleasure, consciousness, intent, or validated learning.
- This project never posts, follows, likes, or otherwise acts on X, and never accepts or stores account credentials. Fixture mode (`--fixture`) is offline and synthetic; a real run uses only locally downloaded, checksum-verified MaleCNS data.
- Persist state, checkpoints and provenance before advancing to another character. Two alternating checkpoint slots keep the last committed state safe during a crash. Refuse resume on any target/decoder/dataset/source mismatch; never silently migrate.
- Do not claim a biological fly is being used, that the network understands language or Michael's X account, or that it independently chose the sentence.
- Keep README short. Detailed model, validation and calibration caveats belong in `docs/`.
