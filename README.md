# CGRL-for-Content-Agnostic-RFFI

## (CL2025 Accept!) Consistency-Guided Robust Learning for Content-Agnostic Radio Frequency Fingerprinting

### File directory description

```
filetree 
├── /dataset
├── util
│  ├── mmd_loss.py
│  ├── CNNmodel_CAM.py
|  └── get_dataset.py
├── /model
├── CAM_Analysis_Tool.py
└── main.py
```

### How to run?

```
python main.py --gpu 0 --len_mark 16 --lam_ACR 0.001 --lam_SCR 0.01 --code_state train_test

python main.py --gpu 0 --len_mark 32 --lam_ACR 0.001 --lam_SCR 0.01 --code_state train_test
```
