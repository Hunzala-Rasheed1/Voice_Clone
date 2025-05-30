# 🎙️ Real-Time Voice Cloning

## 🎯 Overview

This project demonstrates a complete voice cloning pipeline that can:
- **Train voice profiles** from audio samples (5-20 samples)
- **Generate human-like speech** in the trained voice
- **Support 11+ languages** (English, Spanish, French, German, Italian, Portuguese, Russian, Korean, Japanese, Arabic, Polish)
- **Work completely offline** using pyttsx3 TTS engine
- **Real-time processing** with Streamlit web interface

Perfect for YouTube content creators, content localization, accessibility applications, and AI research.

## ✨ Features

### 🎤 Voice Training
- Upload 5-20 voice samples in multiple formats (WAV, MP3, M4A, FLAC)
- Automatic feature extraction using MFCC, pitch analysis, and spectral features
- Quality scoring and voice profile creation
- Support for multiple speakers and languages

### 🗣️ Speech Generation  
- Real-time text-to-speech synthesis
- Voice conversion using trained profiles
- Adjustable speech rate, volume, and cloning strength
- Multiple system voice selection
- Immediate audio playback in browser

### 📊 Analysis & Monitoring
- Voice profile comparison and analysis
- Feature visualization (MFCC, spectral characteristics)
- Quality metrics and embedding analysis
- Generation history and results tracking

### 🌍 Multilingual Support
- **English** (en) - Primary language
- **Spanish** (es) - Full support
- **French** (fr) - Full support  
- **German** (de) - Full support
- **Italian** (it) - Full support
- **Portuguese** (pt) - Full support
- **Russian** (ru) - Full support
- **Korean** (ko) - Full support
- **Japanese** (ja) - Full support
- **Arabic** (ar) - Full support
- **Polish** (pl) - Full support

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/voice-cloning-poc.git
cd voice-cloning-poc

# Create virtual environment
python -m venv venv_voice_clone
source venv_voice_clone/bin/activate  # Windows: venv_voice_clone\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run working_voice_cloning_poc.py
```

### One-Line Install
```bash
pip install streamlit pandas numpy plotly librosa soundfile scipy scikit-learn matplotlib pyttsx3
```

### Launch Application
```bash
streamlit run working_voice_cloning_poc.py
# Open browser to: http://localhost:8501
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 1GB free space
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
librosa>=0.8.0
soundfile>=0.12.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pyttsx3>=2.90
```

### Platform-Specific Setup

**Windows:**
- Works out of the box with Windows SAPI voices
- Includes Microsoft David, Zira, Mark voices

**macOS:**
- Uses built-in macOS voices (Alex, Victoria, Samantha)
- May require Xcode command line tools

**Linux:**
- Requires espeak or festival:
```bash
sudo apt install espeak espeak-data libespeak1 libespeak-dev
# OR
sudo apt install festival festvox-kallpc16k
```

## 📖 Usage Guide

### 1. Voice Training (2-5 minutes)

1. **Navigate to "🎤 Voice Training" tab**
2. **Enter voice profile name** (e.g., "Speaker_1")
3. **Select target language** from 11 supported languages
4. **Upload voice samples**:
   - 5-20 audio files recommended
   - 5-30 seconds each
   - Clear speech, minimal background noise
   - Different sentences/phrases for variety
5. **Click "🚀 Train Voice Profile"**
6. **Wait for processing** (1-3 minutes)

### 2. Speech Generation (10 seconds)

1. **Go to "🗣️ Speech Generation" tab**
2. **Select trained voice profile**
3. **Choose system voice** (optional)
4. **Enter text to synthesize**
5. **Adjust settings**:
   - **Speech Rate**: 50-400 WPM
   - **Volume**: 0.1-1.0
   - **Apply Voice Cloning**: Enable/disable
   - **Cloning Strength**: 0.1-1.0
6. **Click "🎵 Generate Speech"**
7. **Listen to generated audio**

### 3. Analysis & Results

- **📊 Analysis tab**: Compare voice profiles and view characteristics
- **📈 Results tab**: Review generated samples and quality metrics

## 🎵 Audio Guidelines

### Recording Quality
- **Environment**: Quiet room, minimal echo
- **Microphone**: 6-12 inches from mouth  
- **Volume**: Natural speaking level
- **Format**: WAV preferred, MP3/M4A acceptable
- **Content**: Varied sentences, emotions, speaking styles

### Sample Examples
```
sample1.wav: "Hello, my name is [name], and this is a voice sample."
sample2.wav: "The quick brown fox jumps over the lazy dog."
sample3.wav: "How are you doing today? I hope you're having a great day!"
sample4.wav: "Please record this message clearly and naturally."
sample5.wav: "This system will learn my voice characteristics."
```

## 🔧 Architecture

### Voice Processing Pipeline
```
Audio Upload → Feature Extraction → Voice Embedding → Profile Storage
     ↓
MFCC Analysis ← Pitch Detection ← Spectral Analysis ← Quality Scoring
```

### Speech Generation Pipeline
```
Text Input → pyttsx3 TTS → Base Audio → Voice Conversion → Final Output
                                    ↑
                            Voice Profile Features
```

### Technology Stack
- **Frontend**: Streamlit web interface
- **Audio Processing**: librosa, soundfile
- **TTS Engine**: pyttsx3 (offline)
- **Voice Analysis**: MFCC, pitch tracking, spectral analysis
- **Voice Conversion**: Pitch shifting, spectral envelope modification
- **Visualization**: Plotly, matplotlib

## 📊 Performance Metrics

### Expected Results
- **Voice Similarity**: 60-75% match to original
- **Pitch Accuracy**: 80-90% matching
- **Processing Speed**: 2-10 seconds per sentence
- **Training Time**: 1-3 minutes for 10 samples
- **Quality Score**: 0.7+ for good profiles

### Benchmarks
- **Sample Requirements**: 5-20 audio files
- **File Size**: 1-10MB per sample
- **Output Quality**: 22kHz, 16-bit mono WAV
- **Memory Usage**: 100-500MB during processing
