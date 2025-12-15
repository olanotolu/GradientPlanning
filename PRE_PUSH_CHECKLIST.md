# Pre-Push Checklist

## ✅ Ready for GitHub

### Essential Files
- [x] `README.md` - Complete with badges, quickstart, results
- [x] `LICENSE` - MIT License
- [x] `.gitignore` - Properly configured
- [x] `requirements.txt` - All dependencies listed
- [x] `pyproject.toml` - Package metadata
- [x] `setup.py` - Package setup script

### Documentation
- [x] `docs/blog.md` - Technical blog post
- [x] `CONTRIBUTING.md` - Contribution guidelines
- [x] `CODE_OF_CONDUCT.md` - Code of conduct
- [x] `CHANGELOG.md` - Version history
- [x] `PROJECT_STRUCTURE.md` - Project structure
- [x] `RESULTS.md` - Results analysis
- [x] `STATUS.md` - Project status
- [x] `SUMMARY.md` - Complete summary

### GitHub Files
- [x] `.github/workflows/python.yml` - CI workflow
- [x] `.github/ISSUE_TEMPLATE/` - Issue templates
- [x] `.github/PULL_REQUEST_TEMPLATE.md` - PR template

### Code Quality
- [x] All Python files have proper imports
- [x] No `__pycache__` directories
- [x] No `.pyc` files
- [x] Code is functional and tested

### Before First Push

1. **Update README.md**:
   - [ ] Replace `yourusername` with your GitHub username in URLs
   - [ ] Update author info in `pyproject.toml` if desired

2. **Check Large Files**:
   - [ ] `2512.09929v1.pdf` (6.5MB) - Consider excluding or using Git LFS
   - [ ] Check `.gitignore` excludes large data files

3. **Verify .gitignore**:
   - [ ] Data files (`.npz`) are ignored
   - [ ] Checkpoints (`.pt`) are ignored
   - [ ] Results plots are ignored (except demo images)

4. **Test Installation**:
   ```bash
   pip install -r requirements.txt
   python -c "from src.models.world_model import WorldModel; print('OK')"
   ```

## First Commit Command

```bash
git add .
git commit -m "Initial commit: Minimal implementation of gradient-based planning

- Complete 2D navigation environment with wall-door task
- MLP world model with baseline, adversarial, and online training
- Gradient-based and CEM planners
- Evaluation and visualization tools
- Comprehensive documentation and blog post
- Demonstrates train-test gap and 82% error reduction with finetuning"
```

## GitHub Repository Setup

After pushing:

1. **Repository Settings**:
   - Description: "Minimal weekend implementation of 'Closing the Train-Test Gap in World Models for Gradient-Based Planning'"
   - Topics: `world-models`, `gradient-based-planning`, `reinforcement-learning`, `robotics`, `pytorch`, `planning`, `model-based-rl`
   - Enable Issues
   - Enable Discussions (optional)

2. **Create Release**:
   - Tag: `v0.1.0`
   - Title: "Initial Release"
   - Description: See `CHANGELOG.md`

3. **Optional Enhancements**:
   - Add demo images to README
   - Set up GitHub Pages for blog post
   - Add GitHub Actions badges to README

## Files to Exclude (Already in .gitignore)

- `data/*.npz` - Large data files
- `checkpoints/*.pt` - Model checkpoints
- `results/*.png` (except demo images)
- `__pycache__/` - Python cache
- `.DS_Store` - OS files

## Optional: Git LFS for Large Files

If you want to include the PDF or large checkpoints:

```bash
git lfs install
git lfs track "*.pdf"
git lfs track "*.pt"
git add .gitattributes
```

But for a "shitty version", excluding them is fine!

