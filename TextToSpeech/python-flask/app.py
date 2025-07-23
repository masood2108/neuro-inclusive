from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Initialize the recognizer and the paraphrasing model
recognizer = sr.Recognizer()
paraphrase_model = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        audio_data = request.files['audio']
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            improved_text = paraphrase_model(text, max_length=50, num_return_sequences=1)[0]['generated_text']
            return jsonify({"transcribed_text": improved_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
