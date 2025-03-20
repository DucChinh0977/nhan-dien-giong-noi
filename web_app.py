# web_app.py
from flask import Flask, render_template
import os

app = Flask(__name__)

# Route chính để hiển thị giao diện web
@app.route('/')
def index():
    # Đọc kết quả từ file results.txt
    results = []
    if os.path.exists("results.txt"):
        with open("results.txt", "r", encoding="utf-8") as f:
            results = f.readlines()
    return render_template('index.html', results=results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)