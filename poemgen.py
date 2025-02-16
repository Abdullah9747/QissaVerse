import streamlit as st
import torch
import torch.nn.functional as F
import json
import os
from torch import nn
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Urdu Poetry Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with collapsed sidebar
)

# Custom styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .stTitle {
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.5rem !important;
    }
    .poem-container {
        background-color: transparent;
        color: #ffffff;
        padding: 40px;
        border-radius: 15px;
        font-size: 24px;
        direction: rtl;
        text-align: right;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 30px auto;
        max-width: 800px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton > button {
        display: block;
        margin: 20px auto;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
    }
    .input-container {
        max-width: 600px;
        margin: 2rem auto;
        text-align: center;
    }
    .stSlider {
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class GRUPoemGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(GRUPoemGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out, hidden

@st.cache_data
def load_json_data():
    try:
        base_path = Path(__file__).parent
        files = {
            'characters': base_path / 'characters.json',
            'vocab_to_int': base_path / 'vocab_to_int.json',
            'int_to_vocab': base_path / 'int_to_vocab.json'
        }
        
        for name, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {name}.json")
        
        with open(files['characters'], 'r', encoding='utf-8') as f:
            characters = json.load(f)
        with open(files['vocab_to_int'], 'r', encoding='utf-8') as f:
            vocab_to_int = json.load(f)
        with open(files['int_to_vocab'], 'r', encoding='utf-8') as f:
            int_to_vocab_loaded = json.load(f)
            int_to_vocab = {int(k): v for k, v in int_to_vocab_loaded.items()}
            
        return characters, vocab_to_int, int_to_vocab
    
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format in data files: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        raise

def generate_poem(model, seed_text, vocab_to_int, int_to_vocab, seq_length, length, temperature):
    characters = list(seed_text)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _ in range(length):
            if len(characters) < seq_length:
                padding = [''] * (seq_length - len(characters))
                input_chars = padding + characters
            else:
                input_chars = characters[-seq_length:]
            
            input_indices = [vocab_to_int.get(char, 0) for char in input_chars if char in vocab_to_int]
            
            while len(input_indices) < seq_length:
                input_indices.insert(0, 0)
            
            input_seq = torch.tensor([input_indices], dtype=torch.long).to(device)
            
            output, _ = model(input_seq)
            probabilities = F.softmax(output/temperature, dim=-1)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = int_to_vocab[next_char_idx]
            characters.append(next_char)

    return ''.join(characters)

def main():
    st.title("✨ Urdu Poetry Generator ✨")
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #888;'><em>Create beautiful Urdu poetry using artificial intelligence</em></p>", unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown("<h3 style='text-align: center; color: #1E88E5;'>🎮 Generation Settings</h3>", unsafe_allow_html=True)
        st.markdown("---")
        temperature = st.slider(
            "🌡️ Creativity Level",
            0.1, 1.0, 0.5, 0.1,
            help="Higher values create more creative but potentially less coherent poetry"
        )
        st.markdown("---")
        length = st.slider(
            "📏 Poem Length",
            50, 500, 250, 50,
            help="Number of characters to generate"
        )

    try:
        # Load data and initialize model
        characters, vocab_to_int, int_to_vocab = load_json_data()
        
        vocab_size = len(set(characters))
        embedding_dim = 512
        hidden_dim = 256
        n_layer = 2
        seq_length = 102
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        @st.cache_resource
        def load_model():
            try:
                model = GRUPoemGenerator(vocab_size, embedding_dim, hidden_dim, n_layer).to(device)
                model_path = Path(__file__).parent / 'gru_model.pth'
                
                if not model_path.exists():
                    raise FileNotFoundError("Model file 'gru_model.pth' not found")
                    
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                return model
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                raise

        # Input container
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        seed_text = st.text_input(
            "✍️ Enter your starting words:",
            value="ishq ",
            help="Start your poem with a few words"
        )
        generate_button = st.button("✨ Generate Poem ✨")
        st.markdown("</div>", unsafe_allow_html=True)

        if generate_button:
            try:
                model = load_model()
                with st.spinner("🎨 Crafting your masterpiece..."):
                    generated_text = generate_poem(
                        model=model,
                        seed_text=seed_text,
                        vocab_to_int=vocab_to_int,
                        int_to_vocab=int_to_vocab,
                        seq_length=seq_length,
                        length=length,
                        temperature=temperature
                    )
                
                st.markdown("<h3 style='text-align: center; color: #1E88E5;'>📜 Your Generated Poem</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="poem-container">{generated_text}</div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"⚠️ Error generating poem: {str(e)}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        st.info("📁 Please ensure all required files are present in the application directory.")

if __name__ == "__main__":
    main()