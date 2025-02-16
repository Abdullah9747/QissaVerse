import streamlit as st
import torch
import torch.nn.functional as F
import json
from torch import nn

# Page config
st.set_page_config(
    page_title="Urdu Poetry Generator",
    page_icon="📝",
)

# Model class definition
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

# Load data
@st.cache_data
def load_json_data():
    with open('characters.json', 'r', encoding='utf-8') as f:
        characters = json.load(f)
    with open('vocab_to_int.json', 'r', encoding='utf-8') as f:
        vocab_to_int = json.load(f)
    with open('int_to_vocab.json', 'r', encoding='utf-8') as f:
        int_to_vocab_loaded = json.load(f)
        int_to_vocab = {int(k):v for k,v in int_to_vocab_loaded.items()}
    return characters, vocab_to_int, int_to_vocab

def generate_poem(model, seed_text, vocab_to_int, int_to_vocab, seq_length, length, temperature):
    characters = list(seed_text)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for _ in range(length):
            # Handle the case when input is shorter than seq_length
            if len(characters) < seq_length:
                padding = [''] * (seq_length - len(characters))
                input_chars = padding + characters
            else:
                input_chars = characters[-seq_length:]
            
            # Convert characters to indices, handling empty strings
            input_indices = [vocab_to_int.get(char, 0) for char in input_chars if char in vocab_to_int]
            
            # Ensure we have correct sequence length
            while len(input_indices) < seq_length:
                input_indices.insert(0, 0)
            
            input_seq = torch.tensor([input_indices], dtype=torch.long).to(device)
            
            output, _ = model(input_seq)
            probabilities = F.softmax(output/temperature, dim=-1)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = int_to_vocab[next_char_idx]
            characters.append(next_char)

    return ''.join(characters)

# Main app
def main():
    st.title("Urdu Poetry Generator 🎭")
    st.markdown("Generate beautiful Urdu poetry using AI")

    # Load data
    try:
        characters, vocab_to_int, int_to_vocab = load_json_data()
        
        # Model parameters
        vocab_size = len(set(characters))
        embedding_dim = 512
        hidden_dim = 256
        n_layer = 2
        seq_length = 102
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        @st.cache_resource
        def load_model():
            model = GRUPoemGenerator(vocab_size, embedding_dim, hidden_dim, n_layer).to(device)
            checkpoint = torch.load('gru_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model

        # User inputs
        seed_text = st.text_input("Enter seed text:", value="ishq ", help="Start your poem with a few words")
        
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.1, 1.0, 0.5, 0.1, 
                                  help="Higher = more creative, Lower = more focused")
        with col2:
            length = st.slider("Poem Length", 50, 500, 250, 50, 
                             help="Number of characters to generate")

        # Generate button
        if st.button("Generate Poem 🎨"):
            try:
                model = load_model()
                with st.spinner("Crafting your poem..."):
                    generated_text = generate_poem(
                        model=model,
                        seed_text=seed_text,
                        vocab_to_int=vocab_to_int,
                        int_to_vocab=int_to_vocab,
                        seq_length=seq_length,
                        length=length,
                        temperature=temperature
                    )
                
                # Display the generated poem
                st.markdown("### Your Generated Poem:")
                st.markdown(f"""
                <div style='background-color: #f; 
                            padding: 20px; 
                            border-radius: 10px; 
                            font-size: 18px; 
                            direction: rtl; 
                            text-align: right;'>
                    {generated_text}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating poem: {str(e)}")

    except Exception as e:
        st.error(f"Error loading required files: {str(e)}")
        st.info("Please ensure all required model files are present in the application directory.")

if __name__ == "__main__":
    main()