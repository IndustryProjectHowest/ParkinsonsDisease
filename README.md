# Dog Behavior-Based Parkinson’s Detection
A pipeline to detect whether a sample jar (e.g., Parkinson’s odor) is positive or negative based on a dog’s sniffing behavior in short video snippets. We extract “sniff” events via Detectron2, train a GRU-based sequence model on per-frame CNN features, and provide a Streamlit frontend for inference.

## 1. Description

- **Goal**: From a video of a dog investigating sample jars, automatically extract sniff events and classify each as positive/negative based on learned behavior patterns.
- **Pipeline overview**:
  1. **Detect sniff events**: Use Detectron2 to locate nose/jar interactions and group consecutive frames into snippets.
  2. **Feature extraction**: For each snippet, extract per-frame features via a pretrained CNN (e.g., Inception V3 up to pooling layer).
  3. **Sequence modeling**: Feed the feature sequence into a Bidirectional GRU + FC head to predict positive vs negative.
  4. **Frontend (Streamlit)**: Upload a video → run detection → extract snippets → infer with the GRU model → display results.

## 2. Data Analysis
Before training the model, we performed exploratory data analysis on the extracted sniffing snippets to understand the dataset characteristics and guide modeling decisions.
> Positive snippets often have longer average durations, suggesting the dog spends more time on positive samples.  
![alt text](/images/image-4.png)

> In early rounds, the dog’s first sniff may be less accurate; accuracy may improve on revisits, or vice versa.  
![alt text](/images/image-5.png)
## 3. Model Summary

- **Feature extractor**: Pretrained Inception V3 (remove final classification layer) → outputs ~2048-dim vector per frame (resize frames to 299×299, normalize).
- **Sequence model**: Bidirectional GRU:
  - Input size: 2048
  - Hidden size: 128 (example; adjust as needed)
  - Num layers: 1 (or more if desired)
  - Dropout (e.g., 0.3)
  - Output: final hidden state(s) → FC layers → logits for 2 classes.
- **Training**:
  - Loss: CrossEntropyLoss
  - Optimizer: Adam (lr ~1e-4)
  - Batch size: ~16
  - Epochs: ~20–30 (depending on data size)
  - Handle variable-length snippet sequences via padding/packing in DataLoader.
- **Evaluation metrics**: Accuracy, precision/recall/F1 per class, confusion matrix, ROC curve & AUC.
- **Typical results (example placeholders)**:
  - Overall accuracy: ~0.80
  ![alt text](/images/image.png)
  - Positive class: precision ~0.74, recall ~0.66
  - Negative class: precision ~0.83, recall ~0.88
  ![alt text](/images/image-1.png)
  - AUC: ~0.77
  ![alt text](/images/image-2.png)


## 4. Usage
### 4.1 Prerequisites
- Python 3.10+
- GPU recommended (for faster feature extraction & training), but CPU can run inference (slower). be careful with what you installed especially the version you choose, with pure CPU running speed can be extremly slow.
- Virtual environment is recommended.
- Dependencies (in `requirements.txt`):  
  - `torch`, `torchvision`, `opencv-python`, `detectron2` (install per Detectron2 docs), `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `streamlit` (for frontend).

### 4.2 Inference / Frontend
![alt text](/images/image-3.png)
> by running command below after install docker, you can test the model result

```bash
docker pull woyaya114/sniff_predict_app:v2

docker run -d \
  --name sniff_predict_app \
  --restart unless-stopped \
  -p 8501:8501 \
  woyaya114/sniff_predict_app:v2

```
> then open your brower at
```bash
http://localhost:8501