# Attribution and release scope

TAPay adds structured intent, transaction–intent alignment, dynamic decision boundaries, an offline payment environment and evaluation utilities to an adapted hierarchical multi-agent implementation.

The collaboration implementation builds on **TalkHier**, “Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems,” by Zhao Wang, Sota Moriyama, Wei-Yao Wang, Briti Gangopadhyay and Shingo Takamatsu (https://arxiv.org/abs/2502.11098). The upstream distribution supplies Creative Commons Attribution–NonCommercial 4.0 International. Its original license text is retained in `LICENSE-TalkHier.txt` and applies to upstream/derived material. The adapted implementation includes changes to orchestration, structured-output handling, execution controls and audit traces. No endorsement by upstream authors is implied.

This is a minimal source-and-data release, not the full experiment workspace. It excludes literature-baseline implementations, historical runs, raw trajectories, local credentials, provider endpoints, review work files and paper drafts. It therefore does not independently reproduce every comparison table in the manuscript. The mock path uses scripted, label-aware responses for software validation only, never for reporting LLM performance.

No additional license is granted here for original TAPay additions or datasets; their reuse licensing is pending an explicit decision by the rights holder. Public repository access does not override the upstream noncommercial terms.

`model/SOURCE_MANIFEST.json` records source/release hashes. Packaging changes relocate `src` to `model`, update code hashing to that path, add a public command wrapper, and translate code comments/docstrings into English. Internal h1/h2/h3 names are retained for code compatibility and mapped to public Agent names in the README.
