from flask import Flask, request, jsonify, send_file
import torch
from diffusers import DiffusionPipeline
import os
import uuid
import imageio

VIDEO_SERVER_BASE_URL = os.getenv('VIDEO_SERVER_BASE_URL', 'http://localhost:5000')

app = Flask(__name__)

pipe = None

def load_model():
    global pipe
    if pipe is None:
        pipe = DiffusionPipeline.from_pretrained(
            "wanng/wang-t2v",
            torch_dtype=torch.float16
        ).to("cuda")
    return pipe

@app.route('/generate', methods=['POST'])
def generate_video():
    if request.json.get('auth_token') != os.getenv('API_SECRET'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        pipe = load_model()
        video_frames = pipe(prompt, num_inference_steps=30).frames
        
        os.makedirs('videos', exist_ok=True)
        filename = f"{uuid.uuid4().hex}.mp4"
        filepath = os.path.join('videos', filename)
        imageio.mimsave(filepath, video_frames, fps=8)
        
        return jsonify({
    'video_url': f"{VIDEO_SERVER_BASE_URL}/videos/{filename}"
})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/videos/<filename>')
def serve_video(filename):
    return send_file(os.path.join('videos', filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)