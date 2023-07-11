# app.py

from flask import Flask, request, send_file, render_template
from flask_bootstrap import Bootstrap
from main import generate_and_upscale_image

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text_prompt = request.form.get('text_prompt')
        clipdrop_api_key = request.form.get('clipdrop_api_key')
        stability_api_key = request.form.get('stability_api_key')
        replicate_api_token = request.form.get('replicate_api_token')

        _, error = generate_and_upscale_image(text_prompt, clipdrop_api_key, stability_api_key, replicate_api_token)
        if error:
            return render_template('error.html', error=error)
        else:
            return render_template('download.html')

    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True, attachment_filename=filename)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
