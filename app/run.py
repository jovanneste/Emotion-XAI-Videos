from flask import Flask
from flask import render_template

app = Flask(__name__)

# export FLASK_APP=run.py
# export FLASK_ENV=development
# flask run

@app.route("/")
def hello_world():
    return render_template('index.html')
