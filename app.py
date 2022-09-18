from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
 
app = Flask(__name__)
  
CAMINHO = 'C:\Teste'
app.config['CAMINHO'] = CAMINHO
 
EXTENSAO_PERMITIDA = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
def arquivoValido(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSAO_PERMITIDA
 
@app.route('/')
def main():
    return 'Homepage'
 
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'Nenhuma parte do arquivo na solicitação'})
        resp.status_code = 400
        return resp
 
    files = request.files.getlist('files[]')
     
    errors = {}
    success = False
     
    for file in files:
        if file and arquivoValido(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['CAMINHO'], filename))
            success = True
        else:
            errors[file.filename] = 'O tipo de arquivo não é permitido'

    if success:
        resp = jsonify({'message' : 'Upload realizado com sucesso!'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
 
if __name__ == '__main__':
    app.run(debug=True)