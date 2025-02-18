import streamlit as st
import torch
import torch.nn.functional as F
import json
import os
from torch import nn
from pathlib import Path
from gtts import gTTS
from io import BytesIO

st.set_page_config(page_title="Urdu Poetry Generator", page_icon="📝", initial_sidebar_state="collapsed")

class GRUPoemGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(GRUPoemGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

@st.cache_data
def load_json_data():
    base_path = Path(__file__).parent
    files = {name: base_path / f"{name}.json" for name in ['characters', 'vocab_to_int', 'int_to_vocab']}

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {name}.json")

    with open(files['characters'], 'r', encoding='utf-8') as f:
        characters = json.load(f)
    with open(files['vocab_to_int'], 'r', encoding='utf-8') as f:
        vocab_to_int = json.load(f)
    with open(files['int_to_vocab'], 'r', encoding='utf-8') as f:
        int_to_vocab = {int(k): v for k, v in json.load(f).items()}

    return characters, vocab_to_int, int_to_vocab

def generate_poem(model, seed_text, vocab_to_int, int_to_vocab, seq_length, length, temperature):
    characters = list(seed_text)
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(length):
            input_chars = characters[-seq_length:] if len(characters) >= seq_length else [''] * (seq_length - len(characters)) + characters
            input_indices = [vocab_to_int.get(char, 0) for char in input_chars]
            input_seq = torch.tensor([input_indices], dtype=torch.long).to(device)
            output = model(input_seq)
            probabilities = F.softmax(output / temperature, dim=-1)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            characters.append(int_to_vocab[next_char_idx])

    return ''.join(characters)

def synthesize_audio(text):
    try:
        tts = gTTS(text, lang='ur')
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"Error during audio synthesis: {e}")
        return None

def main():
    st.title("✨ Urdu Poetry Generator ✨")
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #888;'><em>Create beautiful Urdu poetry using AI</em></p>", unsafe_allow_html=True)

    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""

    with st.sidebar:
        st.markdown("<h3 style='text-align: center; color: #1E88E5;'>🎮 Generation Settings</h3>", unsafe_allow_html=True)
        temperature = st.slider("🌡️ Creativity Level", 0.1, 1.0, 0.5, 0.1)
        length = st.slider("📏 Poem Length", 50, 500, 250, 50)

    try:
        characters, vocab_to_int, int_to_vocab = load_json_data()
        vocab_size = len(set(characters))
        embedding_dim, hidden_dim, n_layer, seq_length = 512, 256, 2, 102
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        @st.cache_resource
        def load_model():
            model = GRUPoemGenerator(vocab_size, embedding_dim, hidden_dim, n_layer).to(device)
            model_path = Path(__file__).parent / 'gru_model.pth'
            if not model_path.exists():
                raise FileNotFoundError("Model file 'gru_model.pth' not found")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model

        seed_text = st.text_input("✍️ Enter your starting words:", value="ishq ")

        if st.button("✨ Generate Poem ✨"):
            try:
                model = load_model()
                with st.spinner("🎨 Crafting your masterpiece..."):
                    st.session_state.generated_text = generate_poem(model, seed_text, vocab_to_int, int_to_vocab, seq_length, length, temperature)

            except Exception as e:
                st.error(f"Error during poem generation: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")

    # Display generated poem
    if st.session_state.generated_text:
        st.markdown("<h3 style='text-align: center; color: #1E88E5;'>📜 Your Generated Poem:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='poem-container'>{st.session_state.generated_text}</div>", unsafe_allow_html=True)

        # Speak button
        if st.button("🔊 Speak"):
             with st.status("Generating audio...", expanded=True):  # Newer versions of Streamlit
                audio_buffer = synthesize_audio(st.session_state.generated_text)
                if audio_buffer:
                    st.audio(audio_buffer, format='audio/mp3')

if __name__ == "__main__":
    main()
