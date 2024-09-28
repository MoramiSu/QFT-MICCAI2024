# QFT
The official implementation of Design as Desired: Utilizing Visual Question Answering for Multimodal Pre-training. We utilize Visual Question Answering (VQA) for multimodal pre-training to guide the framework focusing on targeted pathological features.We leverage descriptions in medical reports to design multi-granular question-answer pairs associated with different diseases, which assist the framework in pre-training without requiring extra annotations from experts. We also propose a novel pre-training framework with a quasi-textual feature transformer, a module designed to transform visual features into a quasi-textual space closer to the textual domain via a contrastive learning strategy. This narrows the vision-language gap and facilitates modalityalignment.

![image](https://github.com/MoramiSu/QFT/assets/91274335/89de6b63-6392-467b-a6a2-407cd50fd95a)

# Main Results
**Results in Visual Recognition**

![image](https://github.com/user-attachments/assets/d2a01348-ab75-4157-853f-99f300a739b6)

![image](https://github.com/user-attachments/assets/008e3bfc-17df-49e5-92b4-deceb181b1d9)

**Results in Report Generation**

![image](https://github.com/user-attachments/assets/dfde0608-c483-4e68-98fd-9637df213ab5)

![image](https://github.com/user-attachments/assets/bdab2fe5-164d-4c88-aa00-9f3283e27f28)

# Poster
[Poster.pdf](https://github.com/user-attachments/files/17173861/Poster.pdf)

# Implementation
**Setting**
- Set the hyperparameter and path in ./constants.py
**Run training process**
- Run ./models/QFT/QFT_training.py to train the model.
```
python QFT_training.py --gpus 1 --strategy ddp --precision 16 --img_encoder vit_base
```

# Citation
@article{su2024design,  
  title={Design as Desired: Utilizing Visual Question Answering for Multimodal Pre-training},  
  author={Su, Tongkun and Li, Jun and Zhang, Xi and Jin, Haibo and Chen, Hao and Wang, Qiong and Lv, Faqin and Zhao, Baoliang and Hu, Yin},  
  journal={arXiv preprint arXiv:2404.00226},  
  year={2024}  
}

# Acknowledgement
This work was supported by the Guangzhou Science and Technology Program (No. 2023B01J0022), the Key Fundamental Research Program of Shenzhen (No. JCYJ20220818101408019), NSFC General Project (No. 62072452), and the Regional Joint Fund of Guangdong (No. 2021B1515130003, 2021B1515120011).
