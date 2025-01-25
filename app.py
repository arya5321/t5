from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer globally
tokenizer = AutoTokenizer.from_pretrained("arya123321/t5_finetuned")
model = AutoModelForSeq2SeqLM.from_pretrained("arya123321/t5_finetuned")

@app.route("/generate_query", methods=["POST"])
def generate_query():
    """Generate a MongoDB query from natural language input."""
    try:
        # Parse input JSON
        data = request.get_json()
        natural_language_input = data.get("natural_language_input", "")
        max_length = data.get("max_length", 128)

        if not natural_language_input:
            return jsonify({"error": "Missing natural_language_input"}), 400

        # Tokenize the natural language input
        inputs = tokenizer.encode(
            natural_language_input, return_tensors="pt", max_length=512, truncation=True
        )

        # Generate the MongoDB query using the model
        outputs = model.generate(
            inputs, max_length=max_length, num_beams=4, early_stopping=True
        )

        # Decode the generated output into text
        query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"mongodb_query": query})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)
