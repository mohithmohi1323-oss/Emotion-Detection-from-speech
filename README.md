# Emotion Detection from Speech 🎤  

This project focuses on detecting **human emotions from speech** using **Machine Learning techniques**. By analyzing audio features such as **MFCC (Mel-Frequency Cepstral Coefficients)**, the system classifies emotions into categories like **Happy, Sad, Angry, Calm, Fearful, and Neutral**.  
It also provides a **Streamlit Web Application** that allows real-time predictions through microphone input, making the system both interactive and practical.  

---

## 📂 Project Structure  

Emotion-Detection-from-Speech/
│── app.py # Streamlit Web Application
│── train_model.py # Script to train the model
│── sample.py # Script to test audio files
│── requirements.txt # Dependencies
│── models/
│ └── emotion_model.pkl # Trained Model File
│── results.png # Results Image
│──

---

## 📌 Features  
- Detects emotions from speech audio  
- Real-time microphone input support  
- Uses **MFCC feature extraction**  
- Built with **Random Forest Classifier**  
- Web interface created using **Streamlit**  
- Provides instant predictions with a pre-trained model  

--- 

▶️ Usage
Run the Web Application
streamlit run app.py
- Click Record and Analyze

- Speak for a few seconds into your microphone

- The predicted emotion will be displayed on the screen

| Audio File               | Predicted Emotion |
| ------------------------ | ----------------- |
| 03-02-03-01-02-02-24.wav | Happy 😊          |
| 03-02-04-01-01-02-23.wav | Sad 😔            |
| 03-02-05-02-01-01-24.wav | Angry 😡          |
| 03-02-06-02-02-01-23.wav | Fearful 😨 
    
___

📊 Confusion Matrix Results

<img width="1366" height="655" alt="Figure_1" src="https://github.com/user-attachments/assets/1545a430-eab2-4ca7-a6f5-295c2f558e10" />


The confusion matrix above illustrates the performance of the Speech Emotion Detection model across six emotions: angry, calm, fearful, happy, neutral, and sad. Each row represents the actual emotion, while each column represents the predicted emotion.

✅ The diagonal values indicate correct predictions.

❌ Off-diagonal values show where the model confused one emotion with another.

The model achieved high accuracy for emotions like sad (34 correct) and angry (33 correct).

Some misclassifications occurred, such as fearful being confused with angry and happy with fearful.

---

 📊 Results
Sample Results

🎧 03-02-03-01-02-02-24.wav → Happy 😊

🎧 03-02-04-01-01-02-23.wav → Sad 😔

🎧 03-02-05-02-01-01-24.wav → Angry 😡

🎧 03-02-06-02-02-01-23.wav → Fearful 😨

🎧 03-02-01-01-02-02-23.wav → Neutral 😐

🎧 03-02-02-01-02-01-24.wav → Calm 😌

<img width="1245" height="653" alt="Screenshot 2025-08-03 172158" src="https://github.com/user-attachments/assets/69109918-9ef5-414b-bd5d-a278763f75c0" />
- After recorded
<img width="1208" height="657" alt="Screenshot 2025-08-03 172633" src="https://github.com/user-attachments/assets/d74d0b6b-21db-4673-9edc-ad697ca060d6" />

---

🌐 Streamlit Web Application

This project includes a Streamlit web application that allows real-time emotion detection from speech. Through the app, users can record their voice using a microphone, and the system immediately analyzes the audio to predict the corresponding emotion. This makes the model more interactive and user-friendly, providing quick results without running code manually.

Results:
- Audio File
<img width="1320" height="665" alt="Screenshot 2025-08-03 173418" src="https://github.com/user-attachments/assets/6cf48910-8d6f-45f5-97b8-e07430a3e751" />

- Record file
<img width="1355" height="666" alt="Screenshot 2025-08-03 173540" src="https://github.com/user-attachments/assets/7a6dcc87-3c88-4983-969e-c8e349c70887" />

---

📜 License
This project is licensed under the MIT License.

---



