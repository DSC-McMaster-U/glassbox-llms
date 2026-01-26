# Running the Examples

## Quick Setup

### Step 1: Activate Virtual Environment

**Use .venv (install dependencies)**

```bash
source .venv/bin/activate
pip install numpy scikit-learn torch transformers
```

### Step 2: Verify Dependencies

```bash
python -c "import numpy, torch, transformers, sklearn; print('✅ All dependencies available')"
```

### Step 3: Run Examples

From the project root (`/Users/udaykanwar/Developer/glassbox-llms/`):

```bash
# Sentiment classification probe
python -m probes.examples.logistic_sentiment

# Sentiment intensity regression probe
python -m probes.examples.linear_intensity

# Activation space structure analysis (PCA)
python -m probes.examples.pca_structure

# Gender concept direction (CAV)
python -m probes.examples.cav_gender
```
