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

# Try to import gTTS (Google Text-to-Speech)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Try to import pyttsx3 as fallback
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Real Voice Cloning PoC",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language mapping
LANGUAGES = {
    'en': {'name': 'English', 'gtts_lang': 'en'},
    'es': {'name': 'Spanish', 'gtts_lang': 'es'},
    'fr': {'name': 'French', 'gtts_lang': 'fr'},
    'de': {'name': 'German', 'gtts_lang': 'de'},
    'it': {'name': 'Italian', 'gtts_lang': 'it'},
    'pt': {'name': 'Portuguese', 'gtts_lang': 'pt'},
    'ru': {'name': 'Russian', 'gtts_lang': 'ru'},
    'ko': {'name': 'Korean', 'gtts_lang': 'ko'},
    'ja': {'name': 'Japanese', 'gtts_lang': 'ja'},
    'ar': {'name': 'Arabic', 'gtts_lang': 'ar'},
    'pl': {'name': 'Polish', 'gtts_lang': 'pl'},
    'zh': {'name': 'Chinese', 'gtts_lang': 'zh'},
    'hi': {'name': 'Hindi', 'gtts_lang': 'hi'},
    'tr': {'name': 'Turkish', 'gtts_lang': 'tr'},
    'nl': {'name': 'Dutch', 'gtts_lang': 'nl'}
}

# Initialize session state
if 'voice_profiles' not in st.session_state:
    st.session_state.voice_profiles = {}
if 'generated_audio' not in st.session_state:
    st.session_state.generated_audio = []

class VoiceProcessor:
    """Voice processing and feature extraction with fixed librosa compatibility"""
    
    def __init__(self):
        self.sample_rate = 22050
        
    def extract_voice_embedding(self, audio_files):
        """Extract voice embedding from multiple audio samples"""
        try:
            embeddings = []
            
            for audio_file in audio_files:
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
                    audio = np.concatenate(audio_segments)
            except:
                pass  # Keep original if silence removal fails
            
            return audio
            
        except Exception as e:
            st.error(f"Error loading audio: {str(e)}")
            return None
    
    def _compute_speaker_embedding(self, audio):
        """Compute speaker embedding using available librosa features"""
        try:
            features = {}
            
            # MFCC features (most important for voice characteristics)
            try:
                mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
                features['mfcc'] = np.mean(mfccs, axis=1)
                features['mfcc_std'] = np.std(mfccs, axis=1)
            except Exception as e:
                st.warning(f"MFCC extraction failed: {e}")
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

class CloudTTSEngine:
    """Cloud-compatible TTS Engine using gTTS and pyttsx3 fallback"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.tts_engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize TTS engine with fallback options"""
        if GTTS_AVAILABLE:
            st.session_state.tts_engine_type = "gTTS (Google)"
            st.session_state.tts_status = "✅ gTTS Ready"
        elif PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    st.session_state.available_voices = [(voice.id, voice.name) for voice in voices]
                    st.session_state.tts_engine_type = "pyttsx3 (Local)"
                    st.session_state.tts_status = "✅ pyttsx3 Ready"
                else:
                    st.session_state.tts_engine_type = "pyttsx3 (No Voices)"
                    st.session_state.tts_status = "⚠️ pyttsx3 No Voices"
            except Exception as e:
                st.session_state.tts_engine_type = "None"
                st.session_state.tts_status = f"❌ TTS Error: {str(e)}"
        else:
            st.session_state.tts_engine_type = "None"
            st.session_state.tts_status = "❌ No TTS Engine Available"
    
    def get_available_voices(self):
        """Get list of available system voices"""
        if GTTS_AVAILABLE:
            # Return gTTS supported languages
            return [(lang_code, f"gTTS {lang_info['name']}") for lang_code, lang_info in LANGUAGES.items()]
        elif self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices')
                return [(voice.id, voice.name) for voice in voices] if voices else []
            except:
                return []
        return []
    
    def generate_speech(self, text, language='en', voice_id=None, rate=200, volume=1.0, slow=False):
        """Generate speech using available TTS engine"""
        try:
            output_file = os.path.join(tempfile.gettempdir(), f"tts_output_{datetime.now().strftime('%H%M%S')}.wav")
            
            if GTTS_AVAILABLE:
                return self._generate_with_gtts(text, language, output_file, slow)
            elif PYTTSX3_AVAILABLE and self.tts_engine:
                return self._generate_with_pyttsx3(text, voice_id, rate, volume, output_file)
            else:
                st.error("No TTS engine available")
                return None
                
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            return None
    
    def _generate_with_gtts(self, text, language, output_file, slow=False):
        """Generate speech using gTTS"""
        try:
            # Get gTTS language code
            gtts_lang = LANGUAGES.get(language, {}).get('gtts_lang', 'en')
            
            # Create gTTS object
            tts = gTTS(text=text, lang=gtts_lang, slow=slow)
            
            # Save as MP3 first
            mp3_file = output_file.replace('.wav', '.mp3')
            tts.save(mp3_file)
            
            # Convert MP3 to WAV using librosa
            audio, sr = librosa.load(mp3_file, sr=self.sample_rate)
            sf.write(output_file, audio, sr)
            
            # Clean up MP3 file
            if os.path.exists(mp3_file):
                os.unlink(mp3_file)
            
            return output_file if os.path.exists(output_file) else None
            
        except Exception as e:
            st.error(f"gTTS error: {str(e)}")
            return None
    
    def _generate_with_pyttsx3(self, text, voice_id, rate, volume, output_file):
        """Generate speech using pyttsx3"""
        try:
            if voice_id:
                self.tts_engine.setProperty('voice', voice_id)
            
            self.tts_engine.setProperty('rate', rate)
            self.tts_engine.setProperty('volume', volume)
            
            self.tts_engine.save_to_file(text, output_file)
            self.tts_engine.runAndWait()
            
            return output_file if os.path.exists(output_file) else None
            
        except Exception as e:
            st.error(f"pyttsx3 error: {str(e)}")
            return None

def apply_voice_conversion(base_audio_path, voice_embedding, output_path, conversion_strength=0.5):
    """Apply voice conversion to base audio"""
    try:
        # Load base audio
        audio, sr = librosa.load(base_audio_path, sr=22050)
        
        if voice_embedding is not None and len(voice_embedding) > 20:
            converted_audio = audio.copy()
            
            # Apply pitch modification
            if len(voice_embedding) > 46:  # pitch_mean position
                target_pitch = voice_embedding[46]
                if target_pitch > 0:
                    # Calculate pitch shift in semitones
                    pitch_shift_semitones = np.log2(target_pitch / 150.0) * 12  # 150Hz as reference
                    pitch_shift_semitones = np.clip(pitch_shift_semitones, -6, 6)  # Limit range
                    
                    # Apply pitch shifting
                    converted_audio = librosa.effects.pitch_shift(
                        converted_audio, 
                        sr=sr, 
                        n_steps=pitch_shift_semitones * conversion_strength
                    )
            
            # Apply formant shifting (spectral envelope modification)
            if len(voice_embedding) > 40:  # spectral_centroid position
                target_centroid = voice_embedding[40]
                if target_centroid > 0:
                    # Simple spectral tilt adjustment
                    stft = librosa.stft(converted_audio)
                    magnitude = np.abs(stft)
                    phase = np.angle(stft)
                    
                    # Apply frequency-dependent gain
                    freq_bins = magnitude.shape[0]
                    spectral_factor = target_centroid / 2000.0  # Normalize around 2kHz
                    spectral_factor = np.clip(spectral_factor, 0.7, 1.3)
                    
                    for i in range(freq_bins):
                        freq_ratio = i / freq_bins
                        gain = 1.0 + (spectral_factor - 1.0) * freq_ratio * conversion_strength
                        magnitude[i] *= gain
                    
                    # Reconstruct audio
                    modified_stft = magnitude * np.exp(1j * phase)
                    converted_audio = librosa.istft(modified_stft)
            
            # Apply energy/volume adjustment
            if len(voice_embedding) > 48:  # energy position
                target_energy = voice_embedding[48]
                if target_energy > 0:
                    current_energy = np.mean(converted_audio**2)
                    if current_energy > 0:
                        energy_factor = np.sqrt(target_energy / current_energy)
                        energy_factor = np.clip(energy_factor, 0.5, 2.0)
                        converted_audio *= (1.0 + (energy_factor - 1.0) * conversion_strength)
            
            # Normalize to prevent clipping
            if np.max(np.abs(converted_audio)) > 0:
                converted_audio = converted_audio / np.max(np.abs(converted_audio)) * 0.9
            
            # Save converted audio
            sf.write(output_path, converted_audio, sr)
            return True
            
        else:
            # No conversion, just copy original
            audio_normalized = audio / np.max(np.abs(audio)) * 0.9 if np.max(np.abs(audio)) > 0 else audio
            sf.write(output_path, audio_normalized, sr)
            return False
            
    except Exception as e:
        st.error(f"Error in voice conversion: {str(e)}")
        # Fallback: copy original file
        try:
            audio, sr = librosa.load(base_audio_path, sr=22050)
            sf.write(output_path, audio, sr)
            return False
        except:
            return False

def audio_to_base64(file_path):
    """Convert audio file to base64 for playback"""
    try:
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            return audio_base64
    except Exception as e:
        st.error(f"Error converting audio to base64: {str(e)}")
        return None

def main():
    st.title("🎙️ Voice Cloning")
    st.markdown("**Real voice cloning using cloud-compatible TTS engines**")
    
    # Initialize components
    voice_processor = VoiceProcessor()
    tts_engine = CloudTTSEngine()
    
    # Sidebar
    st.sidebar.title("🎛️ Voice Cloning System")
    
    # Show TTS engine status
    engine_status = st.session_state.get('tts_status', '❌ Unknown')
    engine_type = st.session_state.get('tts_engine_type', 'Unknown')
    
    st.sidebar.write(f"**TTS Engine:** {engine_type}")
    if engine_status.startswith('✅'):
        st.sidebar.success(engine_status)
    elif engine_status.startswith('⚠️'):
        st.sidebar.warning(engine_status)
    else:
        st.sidebar.error(engine_status)
    
    # Check available voices
    available_voices = tts_engine.get_available_voices()
    if available_voices:
        st.sidebar.success(f"✅ {len(available_voices)} Voices Available")
    else:
        st.sidebar.warning("⚠️ No voices detected")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Voice Training", "🗣️ Speech Generation", "📊 Analysis", "📈 Results"])
    
    with tab1:
        voice_training_tab(voice_processor)
    
    with tab2:
        speech_generation_tab(tts_engine)
    
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

def speech_generation_tab(tts_engine):
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
            
            # Get language from voice profile
            voice_language = st.session_state.voice_profiles[selected_voice]['language']
        else:
            st.warning("⚠️ Please train a voice profile first!")
            return
        
        # System voice selection (for pyttsx3 only)
        available_voices = tts_engine.get_available_voices()
        if available_voices and not GTTS_AVAILABLE:
            system_voice_options = ["Auto"] + [f"{name} ({voice_id[:30]}...)" for voice_id, name in available_voices]
            selected_system_voice = st.selectbox("System Voice", system_voice_options)
            
            if selected_system_voice != "Auto":
                # Find the corresponding voice_id
                voice_index = system_voice_options.index(selected_system_voice) - 1
                selected_voice_id = available_voices[voice_index][0]
            else:
                selected_voice_id = None
        else:
            selected_voice_id = None
        
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
            if GTTS_AVAILABLE:
                slow_speech = st.checkbox("Slow Speech", value=False)
                rate = 200  # Not used for gTTS
                volume = 1.0  # Not used for gTTS
            else:
                rate = st.slider("Speech Rate", 50, 400, 200, help="Words per minute")
                volume = st.slider("Volume", 0.1, 1.0, 0.9, 0.1)
                slow_speech = False
        
        with col_set2:
            apply_cloning = st.checkbox("Apply Voice Cloning", value=True)
            conversion_strength = st.slider("Cloning Strength", 0.1, 1.0, 0.7, 0.1)
        
        if st.button("🎵 Generate Speech", type="primary") and text_input.strip():
            generate_speech_with_cloning(tts_engine, selected_voice, text_input, 
                                       voice_language, selected_voice_id, rate, volume, 
                                       slow_speech, apply_cloning, conversion_strength)
    
    with col2:
        st.subheader("🎛️ Current Settings")
        
        if selected_voice in st.session_state.voice_profiles:
            profile = st.session_state.voice_profiles[selected_voice]
            st.write(f"**Voice Profile:** {selected_voice}")
            st.write(f"**Language:** {LANGUAGES[profile['language']]['name']}")
            st.write(f"**Quality Score:** {profile['quality_score']:.2f}")
            st.write(f"**Sample Count:** {profile['sample_count']}")
            
            if 'embedding' in profile:
                st.write(f"**Embedding Size:** {len(profile['embedding'])}")
        
        # Show TTS engine info
        st.write(f"**TTS Engine:** {st.session_state.get('tts_engine_type', 'Unknown')}")
        
        # Recent generations
        if st.session_state.generated_audio:
            st.subheader("🎵 Recent Generations")
            for result in st.session_state.generated_audio[-3:]:
                st.write(f"• {result['text'][:30]}...")

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
        st.info("No generated audio samples yet")
        return
    
    st.subheader(f"🎵 Generated Samples ({len(st.session_state.generated_audio)} total)")
    
    # Show last 10 results
    for i, result in enumerate(reversed(st.session_state.generated_audio[-10:])):
        with st.expander(f"Sample {len(st.session_state.generated_audio) - i}: {result['text'][:50]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Text:** {result['text']}")
                st.write(f"**Voice Profile:** {result['voice_profile']}")
                st.write(f"**Generated:** {result['timestamp']}")
                st.write(f"**TTS Engine:** {result.get('tts_engine', 'Unknown')}")
                
                # Audio player
                if result.get('audio_base64'):
                    audio_html = f"""
                    <audio controls style="width: 100%;">
                        <source src="data:audio/wav;base64,{result['audio_base64']}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("Audio not available")
            
            with col2:
                st.metric("Duration", f"{result.get('duration', 0):.1f}s")
                if result.get('rate'):
                    st.metric("Rate", f"{result['rate']} WPM")
                cloned_status = "✅ Yes" if result.get('voice_cloned', False) else "❌ No"
                st.write(f"**Voice Cloned:** {cloned_status}")
                
                # Delete button
                if st.button(f"🗑️ Delete", key=f"delete_audio_{i}"):
                    # Remove from session state
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
            # Calculate quality score based on embedding consistency and sample count
            sample_count = len(uploaded_files)
            base_quality = min(0.9, 0.5 + (sample_count / 20) * 0.4)
            embedding_consistency = 1.0 - (np.std(embedding) / (np.mean(np.abs(embedding)) + 1e-6))
            embedding_consistency = np.clip
