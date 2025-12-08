# A very simple Flask application
# Run it with `flask run`

from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/goodbye')
def goodbye():
    return 'Goodbye, World!'


if __name__ == '__main__':
    app.run(debug=True)

