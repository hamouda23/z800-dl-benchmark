from flask import Flask

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return "Bienvenue Samir"

@app.route('/prodotto/<titre>')
def produit(titre):
    return f'Description de produit : {titre}'

if __name__ == '__main__':
    app.run(debug=True)
