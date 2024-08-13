from flask import Flask, render_template, request
from app.recommend import find_similar_recipes

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        user_input_recipe = request.form['user_input_recipe']
        recommendations = find_similar_recipes(user_input_recipe)
        return render_template('index.html', user_input=user_input_recipe, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)