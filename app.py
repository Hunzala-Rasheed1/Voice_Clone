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

# Try to import pyttsx3
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
    'en': {'name': 'English', 'pyttsx3_voice': 'english'},
    'es': {'name': 'Spanish', 'pyttsx3_voice': 'spanish'},
    'fr': {'name': 'French', 'pyttsx3_voice': 'french'},
    'de': {'name': 'German', 'pyttsx3_voice': 'german'},
    'it': {'name': 'Italian', 'pyttsx3_voice': 'italian'},
    'pt': {'name': 'Portuguese', 'pyttsx3_voice': 'portuguese'},
    'ru': {'name': 'Russian', 'pyttsx3_voice': 'russian'},
    'ko': {'name': 'Korean', 'pyttsx3_voice': 'korean'},
    'ja': {'name': 'Japanese', 'pyttsx3_voice': 'japanese'},
    'ar': {'name': 'Arabic', 'pyttsx3_voice': 'arabic'},
    'pl': {'name': 'Polish', 'pyttsx3_voice': 'polish'}
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

class PyTTSxEngine:
    """TTS Engine using pyttsx3"""
    
    def __init__(self):
        self.engine = None
        self.sample_rate = 22050
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize pyttsx3 engine"""
        try:
            if PYTTSX3_AVAILABLE:
                self.engine = pyttsx3.init()
                
                # Get available voices
                voices = self.engine.getProperty('voices')
                if voices:
                    st.session_state.available_voices = [(voice.id, voice.name) for voice in voices]
                else:
                    st.session_state.available_voices = []
            else:
                st.error("pyttsx3 not available. Please install with: pip install pyttsx3")
        except Exception as e:
            st.error(f"Error initializing TTS engine: {str(e)}")
    
    def get_available_voices(self):
        """Get list of available system voices"""
        if self.engine:
            try:
                voices = self.engine.getProperty('voices')
                return [(voice.id, voice.name) for voice in voices] if voices else []
            except:
                return []
        return []
    
    def generate_speech(self, text, voice_id=None, rate=200, volume=1.0):
        """Generate speech using pyttsx3"""
        try:
            if not self.engine:
                return None
            
            # Create temporary file
            output_file = os.path.join(tempfile.gettempdir(), f"tts_output_{datetime.now().strftime('%H%M%S')}.wav")
            
            # Set voice properties
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Generate speech
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            
            # Check if file was created
            if os.path.exists(output_file):
                return output_file
            else:
                st.error("Failed to generate audio file")
                return None
                
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
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
    st.markdown("**Real voice cloning using pyttsx3 TTS engine**")
    
    # Check pyttsx3 availability
    if not PYTTSX3_AVAILABLE:
        st.error("❌ pyttsx3 not installed! Please run: `pip install pyttsx3`")
        st.stop()
    
    # Initialize components
    voice_processor = VoiceProcessor()
    tts_engine = PyTTSxEngine()
    
    # Sidebar
    st.sidebar.title("🎛️ Voice Cloning System")
    st.sidebar.success("✅ pyttsx3 TTS Ready")
    
    # Check available system voices
    available_voices = tts_engine.get_available_voices()
    if available_voices:
        st.sidebar.success(f"✅ {len(available_voices)} System Voices Found")
    else:
        st.sidebar.warning("⚠️ No system voices detected")
    
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
        else:
            st.warning("⚠️ Please train a voice profile first!")
            return
        
        # System voice selection
        available_voices = tts_engine.get_available_voices()
        if available_voices:
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
            st.warning("No system voices available")
        
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
            rate = st.slider("Speech Rate", 50, 400, 200, help="Words per minute")
            volume = st.slider("Volume", 0.1, 1.0, 0.9, 0.1)
        
        with col_set2:
            apply_cloning = st.checkbox("Apply Voice Cloning", value=True)
            conversion_strength = st.slider("Cloning Strength", 0.1, 1.0, 0.7, 0.1)
        
        if st.button("🎵 Generate Speech", type="primary") and text_input.strip():
            generate_speech_with_cloning(tts_engine, selected_voice, text_input, 
                                       selected_voice_id, rate, volume, apply_cloning, conversion_strength)
    
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
                st.metric("Rate", f"{result.get('rate', 0)} WPM")
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
        else:
            st.error("❌ Failed to extract voice characteristics. Please check audio quality.")
            
    except Exception as e:
        st.error(f"Error training voice profile: {str(e)}")

def generate_speech_with_cloning(tts_engine, voice_name, text, voice_id, rate, volume, apply_cloning, conversion_strength):
    """Generate speech with voice cloning"""
    try:
        with st.spinner("Generating speech..."):
            
            # Generate base speech with pyttsx3
            base_audio_path = tts_engine.generate_speech(text, voice_id, rate, volume)
            
            if not base_audio_path or not os.path.exists(base_audio_path):
                st.error("❌ Failed to generate base speech")
                return
            
            profile = st.session_state.voice_profiles[voice_name]
            final_audio_path = base_audio_path
            voice_cloned = False
            
            # Apply voice cloning if requested and embedding available
            if apply_cloning and 'embedding' in profile:
                cloned_audio_path = os.path.join(tempfile.gettempdir(), f"cloned_{voice_name}_{datetime.now().strftime('%H%M%S')}.wav")
                
                voice_cloned = apply_voice_conversion(
                    base_audio_path, 
                    profile['embedding'], 
                    cloned_audio_path, 
                    conversion_strength
                )
                
                if voice_cloned and os.path.exists(cloned_audio_path):
                    final_audio_path = cloned_audio_path
            
            # Convert to base64 for playback
            audio_base64 = audio_to_base64(final_audio_path)
            
            if audio_base64:
                # Calculate duration
                try:
                    audio, sr = librosa.load(final_audio_path, sr=22050)
                    duration = len(audio) / sr
                except:
                    duration = len(text.split()) * 0.3  # Estimate based on word count
                
                # Store result
                result = {
                    'text': text,
                    'voice_profile': voice_name,
                    'audio_base64': audio_base64,
                    'duration': duration,
                    'rate': rate,
                    'volume': volume,
                    'voice_cloned': voice_cloned,
                    'conversion_strength': conversion_strength if apply_cloning else 0,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.generated_audio.append(result)
                
                st.success("✅ Speech generated successfully!")
                
                # Play audio immediately
                audio_html = f"""
                <audio controls autoplay style="width: 100%;">
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"🕐 Duration: {duration:.1f}s")
                with col2:
                    st.info(f"🎯 Rate: {rate} WPM")
                with col3:
                    cloned_status = "✅ Yes" if voice_cloned else "❌ No"
                    st.info(f"🎭 Voice Cloned: {cloned_status}")
            
            # Clean up temporary files
            try:
                if os.path.exists(base_audio_path):
                    os.unlink(base_audio_path)
                if final_audio_path != base_audio_path and os.path.exists(final_audio_path):
                    os.unlink(final_audio_path)
            except:
                pass
                
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
                st.metric("Embedding Range", f"{np.min(embedding):.3f} to {np.max(embedding):.3f}")
                st.metric("Embedding Mean", f"{np.mean(embedding):.3f}")
                st.metric("Embedding Std", f"{np.std(embedding):.3f}")
        
        # Embedding visualization
        if 'embedding' in profile:
            st.markdown("#### 🎵 Voice Feature Visualization")
            
            embedding = profile['embedding']
            
            # Split embedding into meaningful parts
            mfcc_part = embedding[:20] if len(embedding) >= 20 else embedding
            spectral_part = embedding[40:49] if len(embedding) >= 49 else []
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # MFCC visualization
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
            
            with col_viz2:
                # Spectral features
                if len(spectral_part) > 0:
                    feature_names = ['Centroid', 'Centroid_Std', 'Rolloff', 'ZCR', 'Pitch_Mean', 'Pitch_Std', 'Pitch_Range', 'Energy', 'RMS']
                    feature_names = feature_names[:len(spectral_part)]
                    
                    fig_spectral = go.Figure()
                    fig_spectral.add_trace(go.Bar(
                        x=feature_names,
                        y=spectral_part,
                        name='Spectral Features'
                    ))
                    fig_spectral.update_layout(
                        title="Spectral Features",
                        xaxis_title="Feature",
                        yaxis_title="Value",
                        height=300
                    )
                    fig_spectral.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_spectral, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in voice analysis: {str(e)}")

def show_voice_comparison(selected_profiles):
    """Show comparison between voice profiles"""
    try:
        comparison_data = []
        
        for profile_name in selected_profiles:
            profile = st.session_state.voice_profiles[profile_name]
            
            comparison_data.append({
                'Profile': profile_name,
                'Language': LANGUAGES[profile['language']]['name'],
                'Samples': profile['sample_count'],
                'Quality': f"{profile['quality_score']:.3f}",
                'Embedding Size': len(profile.get('embedding', [])),
                'Created': profile['created']
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Feature comparison radar chart
        if all('embedding' in st.session_state.voice_profiles[p] for p in selected_profiles):
            st.subheader("Voice Characteristic Comparison")
            
            fig = go.Figure()
            
            for profile_name in selected_profiles:
                embedding = st.session_state.voice_profiles[profile_name]['embedding']
                
                # Extract key features for comparison
                if len(embedding) >= 49:
                    features = {
                        'Pitch': embedding[46] / 300.0,  # Normalize pitch
                        'Energy': embedding[47] * 1000,  # Scale energy
                        'Spectral Centroid': embedding[40] / 3000.0,  # Normalize centroid
                        'Spectral Rolloff': embedding[42] / 5000.0,  # Normalize rolloff
                        'MFCC Variance': np.std(embedding[:20]),  # MFCC variability
                    }
                    
                    categories = list(features.keys())
                    values = list(features.values())
                    
                    # Normalize values to 0-1 range for radar chart
                    values = [(v - min(values)) / (max(values) - min(values) + 1e-6) for v in values]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=profile_name
                    ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Voice Profile Comparison",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in comparison: {str(e)}")

if __name__ == "__main__":
    main()
