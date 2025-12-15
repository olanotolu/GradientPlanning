# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-XX

### Added
- Initial implementation of gradient-based planning with world models
- 2D wall-door navigation environment
- MLP world model with residual connections
- Gradient-based planner (GBP) with action parameterization
- Cross-entropy method (CEM) baseline planner
- Baseline training with teacher-forcing MSE
- Adversarial world modeling with FGSM-style perturbations
- Online world modeling with DAgger-style finetuning
- Evaluation scripts with success rate and world model error metrics
- Visualization tools for trajectory comparison
- Demo script showing baseline vs finetuned planning
- Improved planners with expert initialization
- Comprehensive documentation (README, blog post, results analysis)

### Results
- Baseline: 2.49 units avg distance, 0.74 world model error
- Online Finetuned: 2.12 units (15% improvement), 0.13 error (82% improvement)
- Expert Init: 1.03 units (44% improvement), 10% success rate

### Fixed
- Gradient computation issues in GBP planning
- Hyperparameter tuning (action scale, planning horizon)
- Model state management during planning

[0.1.0]: https://github.com/yourusername/gradient-planning/releases/tag/v0.1.0

