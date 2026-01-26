
# Folder Structure & Navigation

```
glassbox/
│
├── models/              # Model backends & wrappers
│
├── instrumentation/     # Capture & manipulation of internal states
│   ├── hooks.py
│   ├── activation_store.py
│   └── patching.py
│
├── primitives/          # General-purpose interpretability tools
│   ├── probes/
│   ├── decompositions/
│   │   ├── sae.py       
│   │   └── pca.py
│   └── attribution/
│
├── representations/     # Stable objects derived from primitives
│   ├── features/
│   ├── circuits/
│   └── atlases/
│
├── experiments/         # Orchestration & pipelines
└── utils/
```