from flask import Flask
from flask import render_template, request, jsonify
from ..model_scripts.flask_final_model import predict_proba
from ..model_scripts.preprocessing import process_body, process_title
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score():
    title = request.form.get("title", "empty_title")
    question = request.form.get("body", "<p>empty_question</p>")
    reputation = request.form.get("reputation", 1, type=int)

    proc_body = process_body(question)
    proc_title = process_title(title)

    q_obj = {"body": question, "title" : title, "reputation" : reputation}

    score = predict_proba(q_obj)[0]

    bad = 1 if score > 0.275 else 0

    return jsonify(BS="%.3f" % score, verdict=bad,
                   proc_body=proc_body, proc_title=proc_title)


