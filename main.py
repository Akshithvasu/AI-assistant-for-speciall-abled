import os
import numpy as np
import torch
import wave
import sounddevice as sd
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM
)
from datasets import load_dataset
from collections import deque, Counter
import time
import io
import tempfile
import subprocess
import sys
import traceback

# Import potentially problematic libraries with error handling
try:
    from IPython.display import display, Audio
except ImportError:
    print("Warning: IPython.display not available. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ipython"])
    from IPython.display import display, Audio

try:
    import cv2
except ImportError:
    print("Warning: cv2 not available. Installing opencv-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

try:
    from gtts import gTTS
except ImportError:
    print("Warning: gtts not available. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gtts"])
    from gtts import gTTS

# Helper function to install dependencies if not already installed
def install_dependencies():
    # List of all required packages
    requirements = [
        "torch", "torchvision", "numpy", "Pillow", "sounddevice", 
        "soundfile", "transformers", "datasets", "faster_whisper", "opencv-python",
        "gtts", "ipython"
    ]
    
    print("Checking and installing dependencies...")
    for package in requirements:
        try:
            module_name = package.split('-')[0]  # Handle opencv-python -> opencv
            if module_name == "ipython":
                module_name = "IPython"  # Handle case sensitivity for IPython
            __import__(module_name)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All dependencies installed successfully!")

# Define the TTS (Text-to-Speech) class using gTTS
class EnglishTTS:
    def __init__(self):
        """Initialize the Google text-to-speech system for English."""
        print("Using Google Text-to-Speech (gTTS)...")
        # No models to load for gTTS as it uses Google's web service

    def text_to_speech(self, text, output_filename="output_speech.wav"):
        """Convert English text to speech using Google TTS."""
        if not text or not text.strip():
            print("Error: Empty text provided")
            return None, None

        print(f"Processing text ({len(text)} characters)...")

        # Split long text into chunks for processing (gTTS has length limitations)
        chunks = self._split_text_into_chunks(text)
        print(f"Split into {len(chunks)} chunks for processing")

        # Use temporary file for merging audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name

        # Check if we need to combine multiple chunks
        if len(chunks) == 1:
            try:
                tts = gTTS(text=chunks[0], lang='en', slow=False)
                tts.save(output_filename)
                print(f"Audio saved to {output_filename}")
                
                # Read the audio data for playback
                audio_data, sample_rate = sf.read(output_filename)
                return audio_data, sample_rate
            except Exception as e:
                print(f"Error generating speech: {e}")
                traceback.print_exc()
                return None, None
        else:
            # For multiple chunks, process each and concatenate
            all_audio_data = []
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_filename = f"chunk_{i}.wav"
                    tts = gTTS(text=chunk, lang='en', slow=False)
                    tts.save(chunk_filename)
                    
                    # Read audio data
                    audio_data, sample_rate = sf.read(chunk_filename)
                    all_audio_data.append(audio_data)
                    
                    # Delete temporary chunk file
                    os.remove(chunk_filename)
                except Exception as e:
                    print(f"Error processing chunk {i}: {e}")
                    traceback.print_exc()
            
            if not all_audio_data:
                print("Error: No audio could be generated")
                return None, None
            
            # Concatenate all audio chunks
            combined_audio = np.concatenate(all_audio_data)
            
            # Save the combined audio
            sf.write(output_filename, combined_audio, sample_rate)
            print(f"Audio saved to {output_filename}")
            
            return combined_audio, sample_rate

    def _split_text_into_chunks(self, text, max_chunk_size=500):
        """Split text into manageable chunks for TTS processing.
        gTTS has a limit of around 2000 characters, but we'll use a smaller
        limit for better processing and to avoid potential issues."""
        import re
        # First try to split by sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        sentences = [s for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If sentence is too long, break it down further
            if len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Split long sentence by punctuation or spaces
                sub_chunks = self._split_long_sentence(sentence, max_chunk_size)
                chunks.extend(sub_chunks)
            elif len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_long_sentence(self, sentence, max_size):
        """Split a long sentence into smaller parts at natural break points."""
        import re
        result = []
        remaining = sentence

        while len(remaining) > max_size:
            # Try to find a good break point
            break_point = max_size

            # Look for punctuation to break at
            punctuation_positions = [m.start() for m in re.finditer(r'[,;:]', remaining[:max_size])]
            if punctuation_positions:
                break_point = punctuation_positions[-1] + 1
            else:
                # Fall back to spaces
                space_positions = [m.start() for m in re.finditer(r'\s', remaining[:max_size])]
                if space_positions:
                    break_point = space_positions[-1] + 1

            # Extract the chunk and trim whitespace
            chunk = remaining[:break_point].strip()
            if chunk:
                result.append(chunk)

            # Update remaining text
            remaining = remaining[break_point:].strip()

        if remaining:
            result.append(remaining)

        return result

# Define the image captioning class
class SimpleImageCaptioner:
    def __init__(self):
        """Initialize a simpler image captioning model."""
        print("Loading Image Captioning model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Using a simpler Vision+Language model that's more compatible
        try:
            self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(self.device)
            print("Image Captioning model loaded successfully!")
        except Exception as e:
            print(f"Error loading image captioning model: {e}")
            traceback.print_exc()
            print("Please make sure you have an internet connection and sufficient disk space.")

    def process_image(self, image_path):
        """Load and process an image from path."""
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            traceback.print_exc()
            return None

    def generate_caption(self, image):
        """Generate a caption for the given image."""
        if image is None:
            return "Could not process the image."

        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )

            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            traceback.print_exc()
            return "Failed to generate a caption for this image."

# Voice-to-Text Transcription Class
class VoiceTranscriber:
    def __init__(self):
        """Initialize the voice transcription system."""
        print("Loading Voice Transcription model...")
        try:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                print("faster_whisper not installed. Installing now...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
                from faster_whisper import WhisperModel
                
            self.model = WhisperModel("large-v1", device="cpu", compute_type="float32")
            print("Voice Transcription model loaded successfully!")
        except Exception as e:
            print(f"Error loading voice transcription model: {e}")
            traceback.print_exc()
            print("Please make sure you have an internet connection and sufficient disk space.")
            
    def record_audio(self, file_path, duration):
        """Record audio from microphone."""
        SAMPLE_RATE = 16000  # Sample rate in Hz
        CHANNELS = 1  # Mono audio
        DTYPE = np.int16  # 16-bit audio format
        
        print(f"Recording for {duration} seconds...")
        try:
            frames = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
            sd.wait()  # Wait until recording is finished
            print("Recording complete.")
            
            # Save to WAV file
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(frames.tobytes())
                
            return file_path
        except Exception as e:
            print(f"Error recording audio: {e}")
            traceback.print_exc()
            return None
        
    def transcribe_audio(self, file_path):
        """Transcribe audio file to text."""
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return ""
            
        print("Transcribing audio...")
        try:
            segments, _ = self.model.transcribe(file_path)
            transcription = " ".join(segment.text for segment in segments)
            
            # Save transcription to file
            with open("transcription_log.txt", "w") as log_file:
                log_file.write(transcription)
            
            print("Transcription saved to transcription_log.txt")
            return transcription
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            traceback.print_exc()
            return ""

# Real-Time Webcam Sign Language Recognition Class
class WebcamSignLanguageRecognizer:
    def __init__(self):
        """Initialize the webcam-based sign language recognition system."""
        print("Setting up Sign Language Recognition...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Define class labels (ASL alphabet + special commands)
        self.class_labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
        
        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize input to model size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        
        # Load model
        try:
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_labels))
            
            # Check if model file exists
            if os.path.exists('resnet18_15fin.pth'):
                self.model.load_state_dict(torch.load('resnet18_15fin.pth', map_location=self.device))
                self.model = self.model.to(self.device)
                self.model.eval()
                print("Sign Language model loaded successfully!")
            else:
                print("Error: Model file 'resnet18_15fin.pth' not found.")
                print("Please download the model file and place it in the current directory.")
                print("This is a custom trained model for ASL recognition.")
        except Exception as e:
            print(f"Error loading sign language model: {e}")
            traceback.print_exc()
        
        # Make sure the output folder exists
        self.save_dir = 'captured_images'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Prediction buffer to smooth out predictions
        self.pred_buffer = deque(maxlen=15)
        self.text_buffer = []  # To collect letters into words
        
    def run_recognition(self):
        """Run the webcam-based sign language recognition system."""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return
                
            print("\n===== Webcam Sign Language Recognition =====")
            print("Place your hand in the blue box and press 'Space' to predict a sign.")
            print("Commands:")
            print("  Space - Capture and predict sign")
            print("  Enter - Add the current prediction to text buffer")
            print("  Tab   - Add a space to text buffer")
            print("  Backspace - Delete last character from text buffer")
            print("  S     - Speak the current text buffer")
            print("  C     - Clear the text buffer")
            print("  Q     - Quit")
            
            box_size = 500  # Size of the capture box
            current_text = ""
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame from webcam")
                    break
                    
                # Flip frame horizontally for more intuitive interaction
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Clamp box size if necessary
                box_size = min(box_size, w, h)
                
                # Calculate box coordinates (centered)
                center_x, center_y = w // 2, h // 2
                x_min = max(center_x - box_size // 2, 0)
                y_min = max(center_y - box_size // 2, 0)
                x_max = min(center_x + box_size // 2, w)
                y_max = min(center_y + box_size // 2, h)
                
                # Draw the capture box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                # Show current text buffer at the bottom
                cv2.putText(frame, f"Text: {current_text}", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show current prediction if available
                if self.pred_buffer:
                    most_common = Counter(self.pred_buffer).most_common(1)[0][0]
                    cv2.putText(frame, f'Prediction: {most_common}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow("ASL Recognition", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Spacebar - capture and predict
                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    
                    # Save image
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    save_path = os.path.join(self.save_dir, f"hand_{timestamp}.png")
                    cv2.imwrite(save_path, hand_roi)
                    print(f"Saved image: {save_path}")
                    
                    # Load and transform
                    img_pil = Image.open(save_path).convert('RGB')
                    input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        label = self.class_labels[predicted.item()]
                    
                    # Add to prediction buffer
                    self.pred_buffer.append(label)
                    most_common = Counter(self.pred_buffer).most_common(1)[0][0]
                    print(f"Prediction: {most_common}")
                
                elif key == 13:  # Enter - add current prediction to text
                    if self.pred_buffer:
                        most_common = Counter(self.pred_buffer).most_common(1)[0][0]
                        if most_common not in ['del', 'nothing', 'space']:
                            current_text += most_common
                            print(f"Added '{most_common}' to text: {current_text}")
                        elif most_common == 'space':
                            current_text += ' '
                            print(f"Added space to text: {current_text}")
                        elif most_common == 'del' and current_text:
                            current_text = current_text[:-1]
                            print(f"Deleted last character: {current_text}")
                
                elif key == 9:  # Tab - add space
                    current_text += ' '
                    print(f"Added space to text: {current_text}")
                
                elif key == 8:  # Backspace - delete last character
                    if current_text:
                        current_text = current_text[:-1]
                        print(f"Deleted last character: {current_text}")
                
                elif key == ord('s'):  # 's' - speak current text
                    if current_text:
                        print(f"Speaking: {current_text}")
                        # Initialize TTS on demand to save resources
                        tts_system = EnglishTTS()
                        audio_data, sample_rate = tts_system.text_to_speech(current_text, "sign_speech.wav")
                        if audio_data is not None:
                            print("Audio generated and saved to sign_speech.wav")
                            # Play audio
                            sd.play(audio_data, sample_rate)
                            sd.wait()
                
                elif key == ord('c'):  # 'c' - clear text buffer
                    current_text = ""
                    print("Text buffer cleared")
                
                elif key == ord('q'):  # 'q' - quit
                    break
            
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            return current_text
        except Exception as e:
            print(f"Error in sign language recognition: {e}")
            traceback.print_exc()
            return ""


def image_to_speech():
    try:
        image_captioner = SimpleImageCaptioner()
        tts_system = EnglishTTS()
        
        print("\n===== Image Options =====")
        print("1. Enter image path")
        print("2. Capture image from webcam")
        print("3. Exit to main menu")
        
        choice = input("\nSelect an option (1-3): ")
        
        if choice == "1":
            # Traditional file path approach
            image_path = input("Enter path to image file: ")
            if not os.path.exists(image_path):
                print(f"Error: File not found: {image_path}")
                return
        
        elif choice == "2":
            # Capture from webcam
            print("Opening webcam. Press SPACE to capture an image, ESC to cancel.")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return
                
            image_path = "captured_image.jpg"
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame from webcam")
                    break
                    
                cv2.imshow("Capture Image (SPACE to capture, ESC to cancel)", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("Canceled image capture")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == 32:  # SPACE key
                    cv2.imwrite(image_path, frame)
                    print(f"Image captured and saved to {image_path}")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == "3":
            print("Returning to main menu.")
            return
        
        else:
            print("Invalid choice. Returning to main menu.")
            return
            
        # Process image
        print(f"Processing image: {image_path}")
        image = image_captioner.process_image(image_path)
        
        if image:
            # Generate caption
            print("Generating caption...")
            caption = image_captioner.generate_caption(image)
            print(f"\nGenerated Caption: {caption}")
            
            # Ask if user wants to edit the caption
            user_input = input("Press Enter to use this caption or type a new one: ")
            final_caption = user_input if user_input.strip() else caption
            
            # Generate speech from caption
            print("\nConverting caption to speech...")
            audio_data, sample_rate = tts_system.text_to_speech(final_caption, "caption_speech.wav")
            
            if audio_data is not None:
                print("Audio generated and saved to caption_speech.wav")
                # Play audio
                sd.play(audio_data, sample_rate)
                sd.wait()
        else:
            print("Failed to process the image")
    except Exception as e:
        print(f"Error in image-to-speech process: {e}")
        traceback.print_exc()

# Function for voice transcription
def voice_to_text():
    try:
        transcriber = VoiceTranscriber()
        
        # Get recording duration
        duration = int(input("Enter recording duration (seconds): "))
        audio_file = "recorded_audio.wav"
        
        # Record audio
        file_path = transcriber.record_audio(audio_file, duration)
        if not file_path:
            print("Error recording audio. Please check your microphone.")
            return ""
        
        # Transcribe audio
        transcription = transcriber.transcribe_audio(audio_file)
        
        print("\nTranscription:")
        print(transcription)
        return transcription
    except Exception as e:
        print(f"Error in voice-to-text process: {e}")
        traceback.print_exc()
        return ""

# Main function - MODIFIED to fix exit functionality
def main():
    print("Starting Multi-Modal Accessibility Assistant...")
    
    # Install dependencies if needed
    try:
        install_dependencies()
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        traceback.print_exc()
        print("Please install the required packages manually and try again.")
        return
    
    while True:
        print("\n===== Multi-Modal Accessibility Assistant =====")
        print("1. Sign Language Recognition (Webcam)")
        print("2. Image to Text to Speech")
        print("3. Voice to Text")
        print("4. Exit")
        
        try:
            choice = input("\nSelect an option (1-4): ")
            
            if choice == "1":
                recognizer = WebcamSignLanguageRecognizer()
                recognizer.run_recognition()
            elif choice == "2":
                image_to_speech()
            elif choice == "3":
                voice_to_text()
            elif choice == "4":
                print("Exiting program.")
                # Explicitly exit the program
                sys.exit(0)
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error processing your choice: {e}")
            traceback.print_exc()
            print("Please try again.")

if __name__ == "__main__":
    main()
