# Breast Cancer Detection with Deep Learning

## 📌 Objective
Developed a deep learning model to classify histopathological images of breast tissue into **cancerous** and **non-cancerous** categories, with a focus on detecting **Invasive Ductal Carcinoma (IDC)**.

---

## 📄 Summary
- Designed and implemented a **Convolutional Neural Network (CNN)** model to automate breast cancer detection from histopathology images.
- Achieved **86% accuracy** on validation dataset with strong class-wise performance:  
  - Precision: **0.89** (non-cancerous)  
  - Precision: **0.77** (cancerous)  
- Applied **data augmentation, dropout, and batch normalization** to improve generalization and reduce overfitting.
- Built an **end-to-end pipeline** including preprocessing, training, evaluation, and feature visualization.

---

## 🛠 Key Responsibilities
- Collected and preprocessed **220K+ histopathology images** from Kaggle IDC dataset.
- Implemented **train-test split** and directory-structured dataset management for scalable training.
- Built a **multi-layer CNN architecture** using separable convolutions, pooling layers, batch normalization, and fully connected layers.
- Applied **image augmentation techniques** (rotation, zoom, shift, shear, flips) to enhance dataset diversity.
- Integrated **model checkpointing, early stopping, and learning rate reduction** for optimized training.
- Conducted **model evaluation** using confusion matrix, classification report, and accuracy/loss plots.
- Implemented **feature map visualization** to interpret CNN filters and activation layers.
- Saved and deployed the trained model for real-time predictions on random test images.

---

## 💻 Technologies & Tools Used
- **Programming Languages:** Python  
- **Libraries & Frameworks:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn, OpenCV  
- **Deep Learning Techniques:** CNN, Separable Convolutions, Batch Normalization, Dropout, Data Augmentation  
- **Optimization & Training:** Adagrad optimizer, Learning Rate Scheduling, Early Stopping, Model Checkpointing  
- **Visualization:** Matplotlib, Seaborn, CNN Feature Maps  
- **Environment:** Anaconda, Jupyter Notebook  

---

## 📚 Background
Breast cancer is the most common form of cancer in women, and **Invasive Ductal Carcinoma (IDC)** is the most common subtype.  
Accurate classification of breast cancer subtypes is a critical clinical task, and **automated deep learning methods** can significantly reduce diagnostic time and error.

📥 **Dataset:** [Kaggle IDC Breast Histopathology Images](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/version/1)

---

## 🖼 Model Performance & Visualizations

### Accuracy & Loss Graphs
![](accu_graph.png)  
![](loss_graph.png)  

### Confusion Matrix
![](cm.png)  

```text
Confusion Matrix
[[36257  3412]
 [ 4370 11466]]

Classification Report
              precision    recall  f1-score   support

           0       0.89      0.91      0.90     39669
           1       0.77      0.72      0.75     15836

    accuracy                           0.86     55505
   macro avg       0.83      0.82      0.82     55505
weighted avg       0.86      0.86      0.86     55505

```


![](conv2d_1_views.png)
![](conv2d_2_views.png)
![](conv2d_3_views.png)
![](cnn_architecture_karan.png)
