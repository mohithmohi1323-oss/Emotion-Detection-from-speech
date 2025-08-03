# Emotion-Detection-from-speech
This project detects human emotions from speech using MFCC feature extraction and a Random Forest Classifier. It can identify emotions such as happy, sad, angry, calm, fearful, and neutral. A Streamlit web app is included for real-time emotion detection through microphone input, making the system easy to use and interactive.

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


---

🤝 Contributing
Contributions are always welcome.
To contribute:

- Fork the repository

- Create a new feature branch

- Commit your changes

- Push to your branch

- Open a Pull Request

---

📜 License
This project is licensed under the MIT License.



