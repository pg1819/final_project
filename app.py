import os

from flask import Flask, render_template, request, Response
from main import detect

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        file = request.files["file"]
        method = request.form["method"]
        file.save(os.path.join("static/uploaded_file/", file.filename))
        return render_template("main.html", download="True", file=file.filename, method=method)

    else:
        return render_template("main.html", download="False", file=None, method=None)


@app.route("/stream/<file>/<method>", methods=["GET"])
def action(file, method):
    response = Response(detect(file, method), mimetype="text/event-stream")
    return response


if __name__ == "__main__":
    app.run()
