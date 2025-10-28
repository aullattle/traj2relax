# Traj2Relax

Diffusion-based structure relaxation model trained on real DFT trajectories.

---

## 🧩 Usage

### Train
```bash
python main.py --mode train
```

### Sample
```bash
python main.py --mode sample --version v493.0
```

---

## 📁 Structure
```
traj2relax/
├── config.yaml
├── data.py
├── main.py
├── metric.py
├── model/
├── noiser.py
├── sample.py
├── train.py
├── traj2relax.py
├── utils.py
└── version.json
```

---

## 📂 Data Format
Data should be stored in LMDB:
```
data/
├── train.lmdb
├── val.lmdb
└── test.lmdb
```

---

## 📄 Example
```bash
# Train
python main.py --mode train

# Sample
python main.py --mode sample --version v493.0
```
