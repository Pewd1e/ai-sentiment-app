from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

history = []

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", history=history)

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Введіть текст"}), 400

    try:

        result = classifier(text)[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)

        entry = {"text": text, "label": label, "score": score}
        history.insert(0, entry)
        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
