from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS with environment variable
allowed_origins = os.getenv('ALLOWED_ORIGINS', '').split(',')
CORS(app, origins=allowed_origins)

# Load DialoGPT tokenizer and model
print("Loading model and tokenizer...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route('/message', methods=['POST'])
def message():
    try:
        data = request.get_json()
        user_message = data['message'].strip()

        # Tokenize with attention_mask
        inputs = tokenizer(
            user_message + tokenizer.eos_token,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1000
        )
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Generate response
        response_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            top_p=0.95,
            top_k=50,
            do_sample=True
        )
        bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return jsonify({'response': bot_response})
    
    except Exception as e:  # ✅ Properly indented within the function
        app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    app.run(host='0.0.0.0', port=port)
