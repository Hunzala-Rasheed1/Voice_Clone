import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import io
import tempfile
import os
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Web Voice Cloning PoC",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language mapping for Web Speech API
LANGUAGES = {
    'en-US': {'name': 'English (US)', 'code': 'en-US'},
    'en-GB': {'name': 'English (UK)', 'code': 'en-GB'},
    'es-ES': {'name': 'Spanish (Spain)', 'code': 'es-ES'},
    'es-MX': {'name': 'Spanish (Mexico)', 'code': 'es-MX'},
    'fr-FR': {'name': 'French', 'code': 'fr-FR'},
    'de-DE': {'name': 'German', 'code': 'de-DE'},
    'it-IT': {'name': 'Italian', 'code': 'it-IT'},
    'pt-BR': {'name': 'Portuguese (Brazil)', 'code': 'pt-BR'},
    'ru-RU': {'name': 'Russian', 'code': 'ru-RU'},
    'ko-KR': {'name': 'Korean', 'code': 'ko-KR'},
    'ja-JP': {'name': 'Japanese', 'code': 'ja-JP'},
    'ar-SA': {'name': 'Arabic', 'code': 'ar-SA'},
    'zh-CN': {'name': 'Chinese (Mandarin)', 'code': 'zh-CN'},
    'hi-IN': {'name': 'Hindi', 'code': 'hi-IN'},
}

# Initialize session state
if 'voice_profiles' not in st.session_state:
    st.session_state.voice_profiles = {}
if 'generated_audio' not in st.session_state:
    st.session_state.generated_audio = []

class VoiceProcessor:
    """Voice processing and feature extraction"""
    
    def __init__(self):
        self.sample_rate = 22050
        
    def extract_voice_embedding(self, audio_files):
        """Extract voice embedding from multiple audio samples"""
        try:
            embeddings = []
            
            for audio_file in audio_files:
                # Reset file pointer
                audio_file.seek(0)
                # Load and preprocess audio
                audio_data = self._load_audio(audio_file)
                if audio_data is not None:
                    # Extract speaker embedding
                    embedding = self._compute_speaker_embedding(audio_data)
                    if embedding is not None:
                        embeddings.append(embedding)
            
            if embeddings:
                # Average embeddings across samples
                avg_embedding = np.mean(embeddings, axis=0)
                return avg_embedding
            else:
                return None
                
        except Exception as e:
            st.error(f"Error extracting voice embedding: {str(e)}")
            return None
    
    def _load_audio(self, audio_file):
        """Load and preprocess audio file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.read())
                tmp_path = tmp_file.name
            
            # Load with librosa
            audio, sr = librosa.load(tmp_path, sr=self.sample_rate)
            
            # Clean up
            os.unlink(tmp_path)
            
            # Normalize and clean
            audio = librosa.util.normalize(audio)
            
            # Remove silence
            try:
                intervals = librosa.effects.split(audio, top_db=20)
                if len(intervals) > 0:
                    audio_segments = []
                    for interval in intervals:
                        audio_segments.append(audio[interval[0]:interval[1]])
                    if audio_segments:
                        audio = np.concatenate(audio_segments)
            except:
                pass  # Keep original if silence removal fails
            
            return audio
            
        except Exception as e:
            st.error(f"Error loading audio: {str(e)}")
            return None
    
    def _compute_speaker_embedding(self, audio):
        """Compute speaker embedding using librosa features"""
        try:
            features = {}
            
            # MFCC features (most important for voice characteristics)
            try:
                mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
                features['mfcc'] = np.mean(mfccs, axis=1)
                features['mfcc_std'] = np.std(mfccs, axis=1)
            except Exception:
                features['mfcc'] = np.zeros(20)
                features['mfcc_std'] = np.zeros(20)
            
            # Spectral features
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
                features['spectral_centroid'] = np.mean(spectral_centroids)
                features['spectral_centroid_std'] = np.std(spectral_centroids)
            except:
                features['spectral_centroid'] = 0
                features['spectral_centroid_std'] = 0
            
            try:
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
                features['spectral_rolloff'] = np.mean(spectral_rolloff)
            except:
                features['spectral_rolloff'] = 0
            
            # Zero crossing rate
            try:
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                features['zcr'] = np.mean(zcr)
            except:
                features['zcr'] = 0
            
            # Pitch estimation
            try:
                pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch_mean'] = np.mean(pitch_values)
                    features['pitch_std'] = np.std(pitch_values)
                    features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
                else:
                    features['pitch_mean'] = 0
                    features['pitch_std'] = 0
                    features['pitch_range'] = 0
            except:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0
            
            # Energy
            try:
                features['energy'] = np.sum(audio**2) / len(audio)
                features['rms_energy'] = np.sqrt(np.mean(audio**2))
            except:
                features['energy'] = 0
                features['rms_energy'] = 0
            
            # Combine all features into embedding
            embedding_parts = [
                features['mfcc'],
                features['mfcc_std'],
                np.array([
                    features['spectral_centroid'],
                    features['spectral_centroid_std'],
                    features['spectral_rolloff'],
                    features['zcr'],
                    features['pitch_mean'],
                    features['pitch_std'],
                    features['pitch_range'],
                    features['energy'],
                    features['rms_energy']
                ])
            ]
            
            embedding = np.concatenate(embedding_parts)
            return embedding
            
        except Exception as e:
            st.error(f"Error computing embedding: {str(e)}")
            return None

def generate_web_speech_js(text, voice_settings, profile_characteristics=None):
    """Generate JavaScript code for Web Speech API with voice modification"""
    
    # Extract voice characteristics for modification
    if profile_characteristics:
        pitch_factor = profile_characteristics.get('pitch_factor', 1.0)
        rate_factor = profile_characteristics.get('rate_factor', 1.0)
        volume_factor = profile_characteristics.get('volume_factor', 1.0)
    else:
        pitch_factor = 1.0
        rate_factor = 1.0
        volume_factor = 1.0
    
    # Apply user settings
    final_pitch = max(0.1, min(2.0, voice_settings['pitch'] * pitch_factor))
    final_rate = max(0.1, min(3.0, voice_settings['rate'] * rate_factor))
    final_volume = max(0.0, min(1.0, voice_settings['volume'] * volume_factor))
    
    js_code = f"""
    <script>
    function speakText() {{
        // Check if speech synthesis is supported
        if ('speechSynthesis' in window) {{
            // Cancel any ongoing speech
            speechSynthesis.cancel();
            
            // Create utterance
            const utterance = new SpeechSynthesisUtterance("{text}");
            
            // Set voice properties
            utterance.lang = "{voice_settings['lang']}";
            utterance.pitch = {final_pitch};
            utterance.rate = {final_rate};
            utterance.volume = {final_volume};
            
            // Try to find specific voice
            const voices = speechSynthesis.getVoices();
            const targetVoice = voices.find(voice => 
                voice.lang === "{voice_settings['lang']}" || 
                voice.lang.startsWith("{voice_settings['lang'][:2]}")
            );
            
            if (targetVoice) {{
                utterance.voice = targetVoice;
            }}
            
            // Set up event handlers
            utterance.onstart = function() {{
                console.log('Speech started');
                document.getElementById('speech-status').innerHTML = '🔊 Speaking...';
            }};
            
            utterance.onend = function() {{
                console.log('Speech ended');
                document.getElementById('speech-status').innerHTML = '✅ Speech completed';
            }};
            
            utterance.onerror = function(event) {{
                console.error('Speech error:', event.error);
                document.getElementById('speech-status').innerHTML = '❌ Speech error: ' + event.error;
            }};
            
            // Speak
            speechSynthesis.speak(utterance);
            
        }} else {{
            document.getElementById('speech-status').innerHTML = '❌ Speech synthesis not supported in this browser';
        }}
    }}
    
    // Auto-load voices
    function loadVoices() {{
        const voices = speechSynthesis.getVoices();
        console.log('Available voices:', voices.length);
    }}
    
    // Load voices when page loads
    if (speechSynthesis.onvoiceschanged !== undefined) {{
        speechSynthesis.onvoiceschanged = loadVoices;
    }}
    loadVoices();
    
    // Auto-speak on load
    setTimeout(speakText, 500);
    </script>
    
    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;">
        <h4>🎵 Speech Output</h4>
        <div id="speech-status">🔄 Initializing speech...</div>
        <br>
        <button onclick="speakText()" style="padding: 8px 16px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer;">
            🔊 Play Speech
        </button>
        <button onclick="speechSynthesis.cancel()" style="padding: 8px 16px; background: #cc0000; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px;">
            ⏹️ Stop
        </button>
    </div>
    """
    
    return js_code

def calculate_voice_characteristics(embedding):
    """Calculate voice modification parameters from embedding"""
    if embedding is None or len(embedding) < 49:
        return None
    
    try:
        # Extract characteristics from embedding
        pitch_mean = embedding[46] if len(embedding) > 46 else 150.0
        energy = embedding[47] if len(embedding) > 47 else 0.1
        spectral_centroid = embedding[40] if len(embedding) > 40 else 2000.0
        
        # Calculate modification factors
        # Pitch factor (relative to average human pitch ~150Hz)
        pitch_factor = np.clip(pitch_mean / 150.0, 0.5, 2.0) if pitch_mean > 0 else 1.0
        
        # Rate factor based on energy (higher energy = faster speech)
        rate_factor = np.clip(1.0 + (energy - 0.1) * 2.0, 0.7, 1.5)
        
        # Volume factor based on RMS energy
        volume_factor = np.clip(0.7 + energy * 3.0, 0.3, 1.0)
        
        return {
            'pitch_factor': pitch_factor,
            'rate_factor': rate_factor,
            'volume_factor': volume_factor,
            'pitch_mean': pitch_mean,
            'energy': energy,
            'spectral_centroid': spectral_centroid
        }
    except Exception as e:
        st.error(f"Error calculating voice characteristics: {e}")
        return None

def main():
    st.title("🎙️ Web Voice Cloning")
    st.markdown("**Voice cloning using browser-based speech synthesis**")
    
    # Browser compatibility notice
    st.info("🌐 This app uses your browser's built-in speech synthesis. Works best in Chrome, Edge, and Safari.")
    
    # Initialize components
    voice_processor = VoiceProcessor()
    
    # Sidebar
    st.sidebar.title("🎛️ Voice Cloning System")
    st.sidebar.success("✅ Web Speech API Ready")
    st.sidebar.info("📱 Works on all modern browsers")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Voice Training", "🗣️ Speech Generation", "📊 Analysis", "📈 Results"])
    
    with tab1:
        voice_training_tab(voice_processor)
    
    with tab2:
        speech_generation_tab()
    
    with tab3:
        analysis_tab()
    
    with tab4:
        results_tab()

def voice_training_tab(voice_processor):
    """Voice training interface"""
    st.header("🎤 Voice Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Voice Samples")
        
        voice_name = st.text_input("Voice Profile Name", placeholder="e.g., MyVoice")
        language = st.selectbox(
            "Target Language", 
            list(LANGUAGES.keys()),
            format_func=lambda x: f"{LANGUAGES[x]['name']} ({x})"
        )
        
        uploaded_files = st.file_uploader(
            "Upload Voice Samples",
            type=['wav', 'mp3', 'm4a', 'flac'],
            accept_multiple_files=True,
            help="Upload 5-20 clear voice samples, each 5-30 seconds long"
        )
        
        if uploaded_files and voice_name:
            st.info(f"📁 {len(uploaded_files)} files uploaded")
            
            if st.button("🚀 Train Voice Profile", type="primary"):
                train_voice_profile(voice_processor, uploaded_files, voice_name, language)
    
    with col2:
        st.subheader("📊 Voice Profiles")
        
        if st.session_state.voice_profiles:
            for name, profile in st.session_state.voice_profiles.items():
                with st.expander(f"✅ {name}"):
                    st.write(f"**Language:** {LANGUAGES[profile['language']]['name']}")
                    st.write(f"**Samples:** {profile['sample_count']}")
                    st.write(f"**Quality:** {profile['quality_score']:.2f}")
                    st.write(f"**Created:** {profile['created']}")
                    
                    # Delete button
                    if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                        del st.session_state.voice_profiles[name]
                        st.rerun()
        else:
            st.info("No voice profiles yet")

def speech_generation_tab():
    """Speech generation interface"""
    st.header("🗣️ Speech Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generate Speech")
        
        # Voice selection
        if st.session_state.voice_profiles:
            selected_voice = st.selectbox(
                "Select Voice Profile",
                list(st.session_state.voice_profiles.keys())
            )
        else:
            st.warning("⚠️ Please train a voice profile first!")
            selected_voice = None
        
        # Language selection
        language = st.selectbox(
            "Speech Language",
            list(LANGUAGES.keys()),
            format_func=lambda x: f"{LANGUAGES[x]['name']} ({x})",
            index=0
        )
        
        # Text input
        text_input = st.text_area(
            "Enter Text to Synthesize",
            height=100,
            placeholder="Type your text here...",
            help="Enter the text you want to convert to speech"
        )
        
        # Generation settings
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            pitch = st.slider("Pitch", 0.1, 2.0, 1.0, 0.1, help="Voice pitch (1.0 = normal)")
            rate = st.slider("Speech Rate", 0.1, 3.0, 1.0, 0.1, help="Speaking speed (1.0 = normal)")
        
        with col_set2:
            volume = st.slider("Volume", 0.0, 1.0, 0.8, 0.1)
            apply_cloning = st.checkbox("Apply Voice Cloning", value=True)
        
        if st.button("🎵 Generate Speech", type="primary") and text_input.strip():
            generate_web_speech(selected_voice, text_input, language, pitch, rate, volume, apply_cloning)
    
    with col2:
        st.subheader("🎛️ Current Settings")
        
        if selected_voice and selected_voice in st.session_state.voice_profiles:
            profile = st.session_state.voice_profiles[selected_voice]
            st.write(f"**Voice Profile:** {selected_voice}")
            st.write(f"**Language:** {LANGUAGES[profile['language']]['name']}")
            st.write(f"**Quality Score:** {profile['quality_score']:.2f}")
            st.write(f"**Sample Count:** {profile['sample_count']}")
            
            if 'embedding' in profile:
                st.write(f"**Embedding Size:** {len(profile['embedding'])}")
                
                # Show voice characteristics if available
                characteristics = calculate_voice_characteristics(profile['embedding'])
                if characteristics:
                    st.write("**Voice Characteristics:**")
                    st.write(f"• Pitch Factor: {characteristics['pitch_factor']:.2f}")
                    st.write(f"• Rate Factor: {characteristics['rate_factor']:.2f}")
                    st.write(f"• Volume Factor: {characteristics['volume_factor']:.2f}")

def analysis_tab():
    """Analysis interface"""
    st.header("📊 Voice Analysis")
    
    if not st.session_state.voice_profiles:
        st.info("No voice profiles to analyze")
        return
    
    profile_names = list(st.session_state.voice_profiles.keys())
    
    # Individual analysis
    st.subheader("📈 Voice Profile Analysis")
    selected_profile = st.selectbox("Select profile to analyze", profile_names)
    
    if selected_profile:
        show_voice_analysis(selected_profile)
    
    # Comparison
    if len(profile_names) >= 2:
        st.subheader("🔍 Compare Voice Profiles")
        selected_profiles = st.multiselect("Select profiles to compare", profile_names, default=profile_names[:2])
        
        if len(selected_profiles) >= 2:
            show_voice_comparison(selected_profiles)

def results_tab():
    """Results interface"""
    st.header("📈 Generation Results")
    
    if not st.session_state.generated_audio:
        st.info("No generated speech samples yet")
        return
    
    st.subheader(f"🎵 Generated Samples ({len(st.session_state.generated_audio)} total)")
    
    # Show last 10 results
    for i, result in enumerate(reversed(st.session_state.generated_audio[-10:])):
        with st.expander(f"Sample {len(st.session_state.generated_audio) - i}: {result['text'][:50]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Text:** {result['text']}")
                st.write(f"**Voice Profile:** {result.get('voice_profile', 'N/A')}")
                st.write(f"**Language:** {result.get('language', 'N/A')}")
                st.write(f"**Generated:** {result['timestamp']}")
                
                # Regenerate speech button
                if st.button(f"🔊 Play Again", key=f"play_again_{i}"):
                    # Regenerate the speech with stored settings
                    voice_settings = {
                        'lang': result.get('language', 'en-US'),
                        'pitch': result.get('pitch', 1.0),
                        'rate': result.get('rate', 1.0),
                        'volume': result.get('volume', 0.8)
                    }
                    
                    profile_characteristics = None
                    if result.get('voice_profile') and result['voice_profile'] in st.session_state.voice_profiles:
                        profile = st.session_state.voice_profiles[result['voice_profile']]
                        if 'embedding' in profile:
                            profile_characteristics = calculate_voice_characteristics(profile['embedding'])
                    
                    js_code = generate_web_speech_js(result['text'], voice_settings, profile_characteristics)
                    st.components.v1.html(js_code, height=150)
            
            with col2:
                st.metric("Pitch", f"{result.get('pitch', 1.0):.1f}")
                st.metric("Rate", f"{result.get('rate', 1.0):.1f}")
                st.metric("Volume", f"{result.get('volume', 0.8):.1f}")
                cloned_status = "✅ Yes" if result.get('voice_cloned', False) else "❌ No"
                st.write(f"**Voice Cloned:** {cloned_status}")
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_audio_{i}"):
                    actual_index = len(st.session_state.generated_audio) - 1 - i
                    st.session_state.generated_audio.pop(actual_index)
                    st.rerun()

def train_voice_profile(voice_processor, uploaded_files, voice_name, language):
    """Train voice profile from uploaded samples"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Processing voice samples...")
        progress_bar.progress(0.2)
        
        # Extract voice embedding
        embedding = voice_processor.extract_voice_embedding(uploaded_files)
        progress_bar.progress(0.8)
        
        if embedding is not None:
            # Calculate quality score
            sample_count = len(uploaded_files)
            base_quality = min(0.9, 0.5 + (sample_count / 20) * 0.4)
            embedding_consistency = 1.0 - (np.std(embedding) / (np.mean(np.abs(embedding)) + 1e-6))
            embedding_consistency = np.clip(embedding_consistency, 0, 1)
            
            quality_score = (base_quality + embedding_consistency) / 2
            quality_score = np.clip(quality_score, 0.1, 0.95)
            
            # Store profile
            st.session_state.voice_profiles[voice_name] = {
                'embedding': embedding,
                'language': language,
                'sample_count': sample_count,
                'quality_score': quality_score,
                'created': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
            progress_bar.progress(1.0)
            status_text.text("✅ Voice profile created successfully!")
            st.success(f"🎉 Voice profile '{voice_name}' created with {sample_count} samples (Quality: {quality_score:.2f})")
            
            # Show voice characteristics
            characteristics = calculate_voice_characteristics(embedding)
            if characteristics:
                st.info(f"🎭 Voice characteristics extracted - Pitch factor: {characteristics['pitch_factor']:.2f}, Rate factor: {characteristics['rate_factor']:.2f}")
        else:
            st.error("❌ Failed to extract voice characteristics. Please check audio quality.")
            
    except Exception as e:
        st.error(f"Error training voice profile: {str(e)}")

def generate_web_speech(voice_name, text, language, pitch, rate, volume, apply_cloning):
    """Generate speech using Web Speech API"""
    try:
        # Voice settings
        voice_settings = {
            'lang': language,
            'pitch': pitch,
            'rate': rate,
            'volume': volume
        }
        
        profile_characteristics = None
        voice_cloned = False
        
        # Apply voice cloning if requested and profile available
        if apply_cloning and voice_name and voice_name in st.session_state.voice_profiles:
            profile = st.session_state.voice_profiles[voice_name]
            if 'embedding' in profile:
                profile_characteristics = calculate_voice_characteristics(profile['embedding'])
                if profile_characteristics:
                    voice_cloned = True
                    st.success(f"🎭 Applying voice characteristics from '{voice_name}'")
        
        # Generate JavaScript for speech
        js_code = generate_web_speech_js(text, voice_settings, profile_characteristics)
        
        # Display the speech interface
        st.components.v1.html(js_code, height=150)
        
        # Store result
        result = {
            'text': text,
            'voice_profile': voice_name if voice_cloned else 'Base Voice',
            'language': language,
            'pitch': pitch,
            'rate': rate,
            'volume': volume,
            'voice_cloned': voice_cloned,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.generated_audio.append(result)
        
        if voice_cloned:
            st.success("✅ Speech generated with voice cloning applied!")
        else:
            st.info("✅ Speech generated with base voice")
            
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")

def show_voice_analysis(profile_name):
    """Show detailed analysis of voice profile"""
    try:
        profile = st.session_state.voice_profiles[profile_name]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Profile Information")
            st.metric("Language", LANGUAGES[profile['language']]['name'])
            st.metric("Sample Count", profile['sample_count'])
            st.metric("Quality Score", f"{profile['quality_score']:.3f}")
            
            # Quality assessment
            quality = profile['quality_score']
            if quality > 0.8:
                st.success("🟢 Excellent voice quality")
            elif quality > 0.6:
                st.warning("🟡 Good voice quality")
            else:
                st.error("🔴 Poor quality - needs more samples")
        
        with col2:
            st.markdown("#### 🎯 Voice Characteristics")
            
            if 'embedding' in profile:
                embedding = profile['embedding']
                st.metric("Embedding Size", len(embedding))
                
                characteristics = calculate_voice_characteristics(embedding)
                if characteristics:
                    st.metric("Pitch Factor", f"{characteristics['pitch_factor']:.3f}")
                    st.metric("Rate Factor", f"{characteristics['rate_factor']:.3f}")
                    st.metric("Volume Factor", f"{characteristics['volume_factor']:.3f}")
                    st.metric("Avg Pitch", f"{characteristics['pitch_mean']:.1f} Hz")
        
        # Embedding visualization
        if 'embedding' in profile:
            st.markdown("#### 🎵 Voice Feature Visualization")
            
            embedding = profile['embedding']
            
            # MFCC visualization
            if len(embedding) >= 20:
                mfcc_part = embedding[:20]
                
                fig_mfcc = go.Figure()
                fig_mfcc.add_trace(go.Scatter(
                    y=mfcc_part,
                    mode='lines+markers',
                    name='MFCC Features',
                    line=dict(color='blue')
                ))
                fig_mfcc.update_layout(
                    title="MFCC Coefficients",
                    xaxis_title="Coefficient Index",
                    yaxis_title="Value",
                    height=300
                )
                st.plotly_chart(fig_mfcc, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in voice analysis: {str(e)}")

def show_voice_comparison(selected_profiles):
    """Show comparison between voice profiles"""
    try:
        comparison_data = []
        
        for profile_name in selected_profiles:
            profile = st.session_state.voice_profiles[profile_name]
            
            characteristics = None
            if 'embedding' in profile:
                characteristics = calculate_voice_characteristics(profile['embedding'])
            
            comparison_data.append({
                'Profile': profile_name,
                'Language': LANGUAGES[profile['language']]['name'],
                'Samples': profile['sample_count'],
                'Quality': f"{profile['quality_score']:.3f}",
                'Pitch Factor': f"{characteristics['pitch_factor']:.3f}" if characteristics else "N/A",
                'Rate Factor': f"{characteristics['rate_factor']:.3f}" if characteristics else "N/A",
                'Created': profile['created']
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Characteristics comparison
        if all('embedding' in st.session_state.voice_profiles[p] for p in selected_profiles):
            st.subheader("Voice Characteristic Comparison")
            
            fig = go.Figure()
            
            for profile_name in selected_profiles:
                profile = st.session_state.voice_profiles[profile_name]
                characteristics = calculate_voice_characteristics(profile['embedding'])
                
                if characteristics:
                    categories = ['Pitch Factor', 'Rate Factor', 'Volume Factor']
                    values = [
                        characteristics['pitch_factor'],
                        characteristics['rate_factor'],
                        characteristics['volume_factor']
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=profile_name
                    ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
                showlegend=True,
                title="Voice Profile Comparison",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in comparison: {str(e)}")

if __name__ == "__main__":
    main()
