from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello Flask"

@app.route("/guest")
def guest():
    name=request.args.get("name", "Guest")
    return f"Hello {name}"
if __name__ == "__main__":
    app.run(debug=True,port=5000,host="0.0.0.0")

#app.run(debug=False, use_reloader=False)


#我是备注