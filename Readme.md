# Oral Path Net
**A deep learning framework for oral pathology detection using convolutional neural networks (CNNs) in TensorFlow / Keras.**

This was a hackathon solution developed for Google Gen AI for Glance pharma problem statement:
> Develop an AI-powered solution that can accurately detect cancer at an early stage using [specific data type, e.g. medical images, genomic data, electronic health records]. The solution should be able to identify high-risk patients and alert healthcare providers for further screening and classify cancer types and stages with high accuracy. The goal is to improve cancer detection rates and enhance patient outcomes through early intervention.

---

## Table of Contents

1. [🚀 Quick Start](#quick-start)  
2. [🧠 Architecture](#architecture)  
3. [📦 Deployment](#deployment)  
4. [✍️ Contributing](#contributing)
5. [📝 Licensing](#licensing)

---

## 🚀 Quick Start
- Via Google Colab (recommended)
It is suggested to train this model via google colab. Just run the notebook linearly as it is , and it should provide results as expected.
If there are any issues please raise a bug here to solve.

- Manual method
```bash
git clone --depth=1 https://github.com/OncoScan/OralPathNet.git
cd OralPathNet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export DATA_DIR=/path/to/images
python train.py \
  --data_dir $DATA_DIR \
  --epochs 50 \
  --batch_size 16 \
  --img_size 224 \
  --model_output ./models/oralpathnet.h5
```

