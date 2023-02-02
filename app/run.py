from flask import Flask

app = Flask(__name__)

# export FLASK_APP=hello.py
# export FLASK_ENV=development
# flask run

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
