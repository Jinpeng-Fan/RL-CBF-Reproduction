# RL-CBF — English academic presentation

- `main.pdf`: final 18-slide presentation, including the title, conclusions and a proposed PhD research plan.
- `main.tex`: editable Beamer source; all equations and diagrams are native LaTeX/TikZ.
- `images/`: vector plots of the final reproduction data for slides 13–16.
- `compile.sh`: run `bash compile.sh` from this directory (requires pdfLaTeX, Beamer, and TikZ).

The visual style follows the supplied IFAC `Conference/Slides/slides.tex`: Madrid theme, default blue-purple palette, rounded theorem blocks, two-line navigation footer, 16:9 format, 10 pt text, and 6 mm side margins. The supplied `18540.pdf` was used as a presentation reference only; its RLHF subject matter is not part of this deck.

The presentation is entirely in English. The sequence is Introduction → Problem Formulation → RL-CBF Method → Simulation Results → Conclusions. There are no repeated outline frames, overlays, or extra appendix pages.

Content source: R. Cheng, G. Orosz, R. M. Murray, J. W. Burdick, “End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks,” AAAI 2019, arXiv:1903.08792v1. Equations are renumbered for the presentation; TeX comments identify the corresponding original equations. The CBF optimization problems use a standard quadratic control/correction objective, 1/2 times the squared Euclidean norm, with a positive linear slack penalty and explicit nonnegative slack. This is an adaptation of the paper's printed unsquared-norm objective, not a claim of identical optimizers. The CBF feasibility constraints and conditional invariance reasoning are retained. Delta u_t is the correction decision variable; Delta u_t^star is the optimizer and equals u_k^CBF(s_t). The executed controller is v_k + u_k^CBF. Vector input bounds apply componentwise. The squared-deviation convention follows Ames et al., Control Barrier Functions: Theory and Applications, ECC 2019, Section II-C: https://coogan.ece.gatech.edu/papers/pdf/amesecc19.pdf. Compensating and guiding control are presented on separate, matched diagram-and-equation slides.

The control architecture follows Efficient implementation. Its editable TikZ diagram distinguishes online signals/data logging (solid) from between-iteration updates (dashed), and shows shared state feedback, RL policy, learned compensation, CBF optimization, environment, GP residual model, and the two logged data sets.


## Reproduction results update

Slides 1–11 review the paper. Their English wording has received light edits, with all mathematical expressions retained. The former Training procedure slide has been removed. Slides 12–17 present the reproduction:

12. Reproduction overview: essential conditions and a brief original-code comparison.
13. Inverted pendulum: training results.
14. Inverted pendulum: closed-loop evaluation before and after training.
15. Car following: training results.
16. Car following: closed-loop evaluation of the trained policies.
17. Conclusions from the reproduction.
18. Research plan: policy optimization and convergence, safety guarantees under model uncertainty, and safe humanoid locomotion.

The method-review slides describe the paper's framework and the QP convention stated above. The result slides use the revised implementation documented in `../report/main.pdf` and `../report/supplement.pdf`. Model, code and tuning details remain in the report.

The four `repro_*.pdf` figures use the same final CSV/NPZ data, 10-episode trailing smoothing and seed mean/min–max aggregation as the report. Their sizes, typography and panel layouts are adapted for projection. `repro_car_policies.pdf` keeps all three response variables from report Figure 4, arranged horizontally.

Data sources relative to the repository root:

- `pendulum/results/final_600ep/<ALG>/seed<k>/episodes.csv`
- `pendulum/results/checkpoint_rerun_seed1/<ALG>/seed1/eval_first_last.npz`
- `car/results/car_final/<ALG>/seed<k>/episodes.csv`
- `car/results/car_eval/sim_seed1.npz`

Regenerate the figures from this directory in PowerShell:

```powershell
& '..\..\.runtime\python37_cpu\python.exe' .\make_reproduction_figures.py
```

Then compile twice using `compile.sh`, or run `pdflatex -output-directory=build main.tex` twice and copy `build/main.pdf` to `main.pdf`.

Return thresholds denote the first crossing of the stated moving average, rather than convergence. Final returns average the last 100 episodes. The car task and reward changed, so returns are compared only within each configuration. Saturation is an observed behavior, without a claim that gradient diagnostics or ablations established its cause.

## Proposed PhD research

The last slide retains the two approved theoretical topics verbatim: policy optimization/convergence under CBF filtering, and safety guarantees/feasibility under model uncertainty. The application is safe humanoid locomotion, informed by CBF-RL (Yang et al., ICRA 2026). The proposed focus is evaluating policies trained with CBFs under terrain, payload and sensing changes, then developing policy verification and lightweight online safeguards. This extends the discussion from observed training/deployment behavior to dependable operation under changed conditions. These are research intentions, not completed results. The visible slide has no bibliography. Frames 1-17 are unchanged.
