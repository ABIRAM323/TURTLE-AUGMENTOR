# app.py
import os
import json
from flask import Flask, request, render_template, send_file, jsonify
from dataset_creator import DatasetCreator
import traceback

app = Flask(__name__)

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/main')
def index():
    return render_template('index.html')

@app.route('/generator')
def generator():
    return render_template('generator.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("Error: No file part in request.")
        return jsonify({'error': 'No file part in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Error: No file selected.")
        return jsonify({'error': 'No file selected.'}), 400

    # Validate file extension
    allowed_extensions = {'.pdf', '.txt', '.csv', '.pptx'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        print(f"Error: Unsupported file extension {file_ext}. Allowed: {allowed_extensions}")
        return jsonify({'error': f'Unsupported file type. Allowed types: {", ".join(allowed_extensions)}'}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        file.save(file_path)
        print(f"File saved successfully: {file_path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

    # Initialize DatasetCreator and process the file
    try:
        creator = DatasetCreator()
        print(f"Processing file: {file_path}")
        extracted_text, text_chunks, dataset = creator.process_files(
            [file_path],
            chunk_size=1500,
            use_rag=True,
            similarity_threshold=0.75,
            in_context_learning=True
        )

        if dataset is None:
            print("Error: Dataset is None after processing.")
            return jsonify({'error': 'Processing failed. Check server logs for details.'}), 500

        # Save dataset to a file
        output_file = 'dataset.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to: {output_file}")

        # Prepare a preview (first 5 QA pairs)
        sample_size = min(5, len(dataset) // 2)
        preview = []
        for i in range(sample_size):
            question = dataset[i * 2]["content"]
            answer = dataset[i * 2 + 1]["content"]
            truncated_answer = answer[:200] + "..." if len(answer) > 200 else answer
            preview.append({'question': question, 'answer': truncated_answer})

        print("File processed successfully. Returning preview.")
        return jsonify({
            'preview': preview,
            'dataset_file': output_file
        })

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()  # Print the full stack trace to the console
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        print(f"Downloading file: {filename}")
        return send_file(filename, as_attachment=True)
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return jsonify({'error': f'Failed to download file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)