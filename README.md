# Vehicle Image Classifier

A computer vision project that classifies vehicle images into 10 categories
(SUV, bus, family sedan, fire engine, heavy truck, jeep, minibus, racing car,
taxi, truck) using transfer learning and fine-tuning with MobileNetV2.

## Results
- **Final validation accuracy: 94.00%**
- Training approach: 10 epochs of classifier-only training, followed by
  5 epochs of fine-tuning the last 4 backbone blocks with a lower learning rate
- No overfitting observed — train/val loss and accuracy curves tracked
  together throughout training (see `training_curves.png`)
- Per-class F1-scores range from 0.90 to 0.98 (see `classification_report.txt`)

## Approach
1. Loaded a MobileNetV2 model pretrained on ImageNet
2. **Phase 1**: Froze the convolutional feature extractor, trained only the
   final classification layer for 10 epochs
3. **Phase 2 (fine-tuning)**: Unfroze the last 4 convolutional blocks and
   continued training for 5 more epochs at a 10x lower learning rate,
   improving validation accuracy from 91.5% to 94.0%
4. Applied data augmentation (random flips, rotation, color jitter) on
   training data to reduce overfitting and improve generalization
5. Saved the best-performing checkpoint (by validation accuracy) rather than
   simply the last epoch

## Analysis
The confusion matrix and per-class report show consistently strong
performance across all 10 classes. The main remaining confusion is between
visually similar vehicle types (e.g., minibus vs. taxi, heavy truck vs. truck),
which is expected given their shared silhouette.

## Demo
A Streamlit web app (`app.py`) lets you upload an image and get a live
prediction with a confidence score.

## Project structure
- `train.py` — training pipeline (data loading, augmentation, training,
  fine-tuning, evaluation, plots)
- `utils.py` — shared model loading and inference logic
- `predict.py` — command-line single-image prediction
- `app.py` — Streamlit web demo

## Dataset
[Vehicle Classification Dataset on Kaggle](https://www.kaggle.com/datasets/marquis03/vehicle-classification)

## How to run
\`\`\`
pip install -r requirements.txt
python train.py       # trains the model and saves the best checkpoint
python predict.py     # run inference on a single test image
streamlit run app.py  # launches the interactive demo
\`\`\`