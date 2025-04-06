from flask import Flask, request, jsonify, render_template
import pandas as pd
from fuzzywuzzy import process

app = Flask(__name__)

# Load dataset
file_path = "climate_fever_dataset.csv"
df = pd.read_csv(file_path)


# Function to classify sentiment and retrieve health impact
def get_sentiment_and_impact(tweet_text):
    result = process.extractOne(tweet_text, df["tweet"].tolist())
    if result:
        match, score = result[0], result[1]
        if score > 80:  # High similarity threshold
            row = df[df["tweet"] == match].iloc[0]
            sentiment_value = row["sentiment_value"]
            health_impact = row["health_impact"]

            if sentiment_value > 0.7:
                sentiment = "Positive"
            elif sentiment_value < 0.3:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            return sentiment, health_impact
    return "Neutral", "No relevant health impact found"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    tweet_text = data.get("tweet", "").strip()

    # Get sentiment and health impact
    sentiment, health_impact = get_sentiment_and_impact(tweet_text)

    return jsonify({"sentiment": sentiment, "health_impact": health_impact})


if __name__ == "__main__":
    app.run(debug=True)
