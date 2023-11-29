from flask import Flask, render_template, request, jsonify
from utils import model_predict, get_encoder_class
app = Flask(__name__)


@app.route("/")
def home():
    category_labels,branch_labels = get_encoder_class()
    return render_template("index.html", category_labels = category_labels, branch_labels =branch_labels)

@app.route('/predict', methods=['POST'])
def predict():
    category_labels,branch_labels = get_encoder_class()

    form_data = request.form.to_dict()
    # nhận diện giá bằng model đã train
    branch_selected = request.form['branch']
    category_selected = request.form['category']
    prediction = model_predict(form_data)
    return render_template("index.html", 
                            form_data=form_data,
                            prediction=prediction, 
                            branch_selected=branch_selected,
                            category_selected=category_selected,
                            category_labels = category_labels, 
                            branch_labels =branch_labels)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
