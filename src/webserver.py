from flask import Flask, render_template, request, jsonify, make_response
from ai_service import openai_chat

app = Flask(__name__)

@app.route("/")   
def home():
    return render_template("index.html")

@app.route("/generate", methods=['POST'])
def get_bot_response():
    data = request.json
    user_text = data.get('msg', '')

    if user_text:
        response = openai_chat(user_text)
    else:
        response = "Please ask a question."

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
