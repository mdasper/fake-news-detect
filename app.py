from flask import Flask, request, render_template
import pickle
from markupsafe import escape

# Load the vectorizer and the model
vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

# Initialize the Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])  # Get news input from the form
        print(news)

        # Make a prediction using the model
        predict = model.predict(vector.transform([news]))[0]
        print(predict)

        # Convert the numeric prediction to a meaningful label
        prediction_label = "Fake" if predict == 0 else "Real"

        # Render the result in the prediction template
        return render_template("prediction.html", prediction_text=f"This News is -> {prediction_label}")
    
    # If not a POST request, just render the prediction page
    else:
        return render_template("prediction.html")

# Run the app
if __name__ == '__main__':
    app.debug = True  # Enable debug mode for development
    app.run()
