# CGRL-for-Content-Agnostic-RFF

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

### Training and testing logs

```
/log
```

### Dataset

```
https://pan.baidu.com/s/1XeH3uMbuwOuYVfePFGBO2A?pwd=mdgh

or 

https://drive.google.com/drive/folders/1Z5iIuYZP2ilej3BsFaOFpx0Go-LOPMZM?usp=sharing


The code for dataset generation is provided in the archive file ``Dataset_SNR_Content_Independent.rar,'' with reference to and thanks to [WTI-Cyber-Team](https://github.com/WTI-Cyber-Team/Public_Wireless_Signal_Datasets)
```

### Requirement

```
torch 1.11.0+cu113

torchaudio 0.11.0+cu113

torchinfo 1.8.0

torchsummary 1.5.1

torchvision 0.12.0+cu113

python 3.8.5

```

### License / 许可证

```
本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
```
