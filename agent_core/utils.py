import os
from django.core.files.storage import FileSystemStorage
import uuid # For generating unique filenames
import time # Import the time module for sleep
import io # Added for gTTS
import json # Added for RAG/LLM (though primarily used in views, good to have if any JSON ops happen here)
import base64 # Added for RAG/LLM (though primarily used in views, good to have if any base64 ops happen here)
import requests # Added for downloading audio

# Langchain imports for PDF processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from django.views.decorators.csrf import csrf_exempt
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# faster-whisper imports for STT
from faster_whisper import WhisperModel

# pyttsx3 import for TTS
import pyttsx3

# Selenium imports for WebDriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException

# Transformers imports for GPT2
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# gTTS import for TTS (used in views but defined here for consistency if needed)
from gtts import gTTS

# Global variables for LLM
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
print("GPT2 Tokenizer and Model initialized.")

# Initialize embeddings model globally for RAG
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print("HuggingFaceEmbeddings initialized for RAG.")

# Directory where FAISS indices are saved
FAISS_INDEX_DIR = './faiss_index'
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
print(f"FAISS_INDEX_DIR ensured: {FAISS_INDEX_DIR}")


#
# Load the Whisper model globally or on first use
try:
    # Using stt_model_dir as the model path as requested by the user
    stt_model = WhisperModel("base", device="cpu", compute_type="int8")
    print(f"Faster Whisper model loaded successfully from ") # Escape curly braces
except Exception as e:
    stt_model = None
    print(f"Error loading Faster Whisper model from ")

# Initialize pyttsx3 engine globally
try:
    tts_engine = pyttsx3.init()
    print("pyttsx3 engine initialized successfully.")
except Exception as e:
    tts_engine = None
    print(f"Error initializing pyttsx3 engine: {e}. TTS functionality will be limited.") # Escape curly braces

def speech_to_text(audio_path):
    """Converts spoken audio from a file to text using Faster Whisper."""
    print(f"Performing STT on: {audio_path}...") # Escape curly braces
    if stt_model and os.path.exists(audio_path):
        try:
            segments, info = stt_model.transcribe(audio_path, beam_size=5)
            recognized_text = " ".join([segment.text for segment in segments])
            print("STT successful.")
            return recognized_text
        except Exception as e:
            print(f"Error during Faster Whisper transcription: {e}") # Escape curly braces
            return ""
    else:
        if not stt_model:
            print("STT model not loaded. Returning dummy text.")
        elif not os.path.exists(audio_path):
            print(f"Audio file not found at {audio_path}. Returning dummy text.") # Escape curly braces
        return "This is a dummy text from speech to text."
def text_to_speech(text_data, output_dir='./tts_audio'):
    """Converts text to speech and saves it to an audio file using pyttsx3."""
    print(f"Performing TTS for text: '{text_data[:50]}'...") # Escape curly braces
    if tts_engine:
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created TTS audio directory: {output_dir}") # Escape curly braces

            filename = f"tts_output_{uuid.uuid4().hex}.mp3"
            audio_filepath = os.path.join(output_dir, filename)

            tts_engine.save_to_file(text_data, audio_filepath)
            tts_engine.runAndWait()
            print(f"TTS successful. Audio saved to: {audio_filepath}") # Escape curly braces
            return audio_filepath
        except Exception as e:
            print(f"Error during pyttsx3 TTS: {e}") # Escape curly braces
            return None
    else:
        print("pyttsx3 engine not initialized. Cannot perform TTS.")
        return None

def handle_uploaded_pdf(uploaded_file):
    """Saves the uploaded PDF file to a local directory."""
    pdf_docs_dir = './pdf_docs'
    if not os.path.exists(pdf_docs_dir):
        os.makedirs(pdf_docs_dir)
        print(f"Created directory: {pdf_docs_dir}") # Escape curly braces

    fs = FileSystemStorage(location=pdf_docs_dir)
    filename = fs.save(uploaded_file.name, uploaded_file)
    file_path = os.path.join(pdf_docs_dir, filename)
    print(f"Saved PDF file to: {file_path}") # Escape curly braces
    return file_path

def process_pdf_to_vectorstore(pdf_path, embeddings_model, chunk_size=1000, chunk_overlap=200):
    """Processes a PDF, chunks it, embeds it, and stores it in a FAISS vector store."""
    print(f"Processing PDF: {pdf_path}") # Escape curly braces

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split PDF into {len(texts)} chunks.") # Escape curly braces

    # Debug print for chunks
    for i, t in enumerate(texts[:3]): # Print first 3 chunks
        print(f"Chunk {i}: {t.page_content[:200]}...") # Escape curly braces

    # Create and save FAISS vector store
    faiss_index_dir = './faiss_index'
    if not os.path.exists(faiss_index_dir):
        os.makedirs(faiss_index_dir)
        print(f"Created directory: {faiss_index_dir}") # Escape curly braces

    if texts:
        vectorstore = FAISS.from_documents(texts, embeddings_model)
        index_path = os.path.join(faiss_index_dir, os.path.basename(pdf_path).replace('.pdf', '_faiss_index.faiss'))
        vectorstore.save_local(index_path)
        print(f"FAISS vector store created and saved to: {index_path}") # Escape curly braces
        return index_path
    else:
        print("No texts to process from PDF.")
        return None

def initialize_webdriver():
    """Initializes and configures the Selenium WebDriver for headless Chrome."""
    print("Initializing Selenium WebDriver...")
    options = Options()
    options.add_argument("--headless=new")  # Run in headless mode (preferred for newer Chrome)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=9222") # For potential debugging

    # Create a user data directory for persistent sessions
    user_data_dir = './user_data/chrome'
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir, exist_ok=True)
        print(f"Created Chrome user data directory: {user_data_dir}")
    options.add_argument(f"user-data-dir={user_data_dir}")

    # Install and get the path to the ChromeDriver executable
    driver_path = ChromeDriverManager().install()
    service = Service(executable_path=driver_path)

    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    print("Selenium WebDriver initialized successfully.")
    return driver

def download_whatsapp_audio(audio_element, driver, temp_dir='./temp_audio_downloads'):
    """Downloads the audio associated with a WhatsApp audio message element."""
    print("Attempting to download WhatsApp audio...")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    audio_source_url = None
    try:
        # Try to find the audio tag and its src attribute directly
        audio_tag = audio_element.find_element(By.TAG_NAME, 'audio')
        audio_source_url = audio_tag.get_attribute('src')
    except Exception as e:
        print(f"Could not find direct audio tag src: {e}")
        # Fallback: Try to find a download button or direct link if the player is complex
        # This part might need further inspection of WhatsApp Web's DOM
        try:
            # Example: look for a common download icon/button
            download_button = audio_element.find_element(By.CSS_SELECTOR, "[data-testid='audio-download']")
            if download_button:
                # Click the download button, which might initiate a direct download
                # or expose a downloadable link. This is highly speculative.
                print("Found potential download button. Attempting click.")
                download_button.click()
                # Wait briefly for potential download to start or URL to appear
                time.sleep(2)
                # In a real scenario, you'd need to monitor network requests or look for new elements
                # For now, we'll assume direct src is the primary method.
        except Exception as e_btn:
            print(f"Could not find or click download button: {e_btn}")

    if not audio_source_url:
        print("No downloadable audio source URL found within the element.")
        return None

    try:
        session = requests.Session() # Create a session for connection reuse
        with session.get(audio_source_url, stream=True) as response:
            response.raise_for_status()

            file_extension = '.ogg' # WhatsApp audio often uses OGG
            if 'content-type' in response.headers and 'mp4' in response.headers['content-type']:
                file_extension = '.mp4'
            elif 'content-type' in response.headers and 'mpeg' in response.headers['content-type']:
                file_extension = '.mp3'

            filename = f"whatsapp_audio_{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(temp_dir, filename)

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"Audio downloaded successfully to: {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading audio from {audio_source_url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during audio download: {e}")
        return None


def get_rag_llm_response(user_query):
    """Encapsulates the RAG and LLM inference logic."""
    context = ""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    try:
        faiss_dirs = [d for d in os.listdir(FAISS_INDEX_DIR) if os.path.isdir(os.path.join(FAISS_INDEX_DIR, d))]

        if faiss_dirs:
            faiss_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(FAISS_INDEX_DIR, x)), reverse=True)
            latest_index_folder = os.path.join(FAISS_INDEX_DIR, faiss_dirs[0])

            print(f"Loading FAISS from: {latest_index_folder}")

            vectorstore = FAISS.load_local(
                latest_index_folder,
                embeddings,
                allow_dangerous_deserialization=True
            )

            docs = vectorstore.similarity_search(user_query, k=2)
            context = "\n\n".join([doc.page_content for doc in docs])

            print(f"Retrieved {len(docs)} document chunks from FAISS for RAG. Context preview: {context[:300]}")

        else:
            print("No FAISS index found for RAG.")

    except Exception as e:
        print(f"Error during FAISS retrieval in get_rag_llm_response: {e}")

    if not context.strip():
        return "No context found from PDF. Please upload PDF first. Can I help with anything else?"

    rag_prompt = f"""
Context:
{context}

Question:
{user_query}

Answer only from the context:
"""

    try:
        inputs = tokenizer(rag_prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            temperature=0.5
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in response_text:
         response_text = response_text.split("Answer:")[-1].strip()

        print("Inference successful with GPT-2.")
        return response_text

    except Exception as e:
        print(f"Error during GPT-2 inference in get_rag_llm_response: {e}")
        return f"Error generating response: {e}"


def monitor_whatsapp_messages(driver):
    """Continuously monitors for new incoming WhatsApp messages and extracts their content."""
    print("Starting WhatsApp message monitoring...")
    processed_message_ids = set()

    while True:
        try:
            print("[WhatsApp Monitor] Checking for new messages...")
            # Find all message elements. The selector might need adjustment based on WhatsApp Web's latest structure.
            # A more robust selector for message bubbles that typically contain data-id
            messages = driver.find_elements(By.CSS_SELECTOR, "div[data-id][data-testid^='message-stub-']") # Targeting elements with data-id and a specific testid pattern
            print(f"[WhatsApp Monitor] Found {len(messages)} message elements.")

            # Process new messages, iterating in reverse to get newest first
            for msg_element in reversed(messages):
                try:
                    message_id = msg_element.get_attribute('data-id')
                    if not message_id:
                        print(f"[WhatsApp Monitor] Skipping message element without data-id: {msg_element.get_attribute('outerHTML')[:200]}...") # Log partial HTML for debugging
                        continue
                    if message_id in processed_message_ids:
                        # print(f"[WhatsApp Monitor] Message {message_id} already processed. Skipping.")
                        continue

                    sender = "Unknown Sender"
                    message_text = ""
                    message_type = "TEXT"
                    user_input_text = ""
                    ai_response_audio_path = None

                    # Attempt to extract sender (e.g., from a name element within the message if available)
                    try:
                        # This selector might need to be relative to msg_element for better accuracy
                        sender_element = msg_element.find_element(By.CSS_SELECTOR, "span[data-testid='conversation-info-header']") # Example selector
                        if sender_element: # More robust sender detection is needed here for real-world use
                            sender = sender_element.text
                    except:
                        pass

                    # Extract text content
                    try:
                        # Adjust selector for text content within a message element
                        text_span = msg_element.find_element(By.CSS_SELECTOR, "span.selectable-text, div[data-pre-plain-text], div.copyable-text span")
                        message_text = text_span.text
                        message_type = "TEXT"
                        print(f"[WhatsApp Monitor] Detected TEXT message with content: {message_text[:50]}...")
                    except:
                        print("[WhatsApp Monitor] No text content found for this message.")
                        pass

                    # Determine if it's an audio message
                    audio_element = None
                    try:
                        # Adjust selector for audio element within a message element
                        audio_element = msg_element.find_element(By.CSS_SELECTOR, "div[data-testid='audio-play-button'], div[data-testid='audio-download'], div[role='button'][aria-label='Play']") # Updated selector for audio detection
                        if audio_element: # If audio element is found
                            message_type = "AUDIO"
                            message_text = "(Audio Message - processing)"
                            print("[WhatsApp Monitor] Detected AUDIO message.")
                    except:
                        print("[WhatsApp Monitor] No audio element found for this message.")
                        pass

                    print(f"[WhatsApp Monitor] Raw Message - ID: {message_id}, From: {sender}, Type: {message_type}, Content: {message_text[:100]}...")

                    if message_type == "AUDIO":
                        print(f"[WhatsApp Monitor] Processing audio message {message_id}...")
                        audio_file_path = download_whatsapp_audio(audio_element, driver)
                        if audio_file_path:
                            user_input_text = speech_to_text(audio_file_path)
                            os.remove(audio_file_path) # Clean up temporary audio file
                            print(f"[WhatsApp Monitor] STT result for {message_id}: {user_input_text[:100]}...")
                        else:
                            user_input_text = "User sent an audio message that could not be processed."
                            print(f"[WhatsApp Monitor] Audio download failed for {message_id}. Using default message for RAG.")
                    else: # TEXT message
                        user_input_text = message_text

                    if user_input_text.strip():
                        print(f"[WhatsApp Monitor] Generating RAG/LLM response for user input: {user_input_text[:50]}...")
                        # Call RAG/LLM to get response
                        ai_response_text = get_rag_llm_response(user_input_text)
                        print(f"[WhatsApp Monitor] AI Response Text: {ai_response_text[:100]}...")

                        # --- Start: Send AI Response via Selenium ---
                        try:
                            print("[WhatsApp Monitor] Attempting to send reply via Selenium...")
                            # Locate the message input field (chatbox)
                            # Common selectors for chat input field (adjust if WhatsApp Web updates)
                            input_field = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='10']"))
                            )
                            input_field.send_keys(ai_response_text)
                            print(f"[WhatsApp Monitor] Typed AI response into chat: {ai_response_text[:50]}...")

                            # Locate and click the send button
                            # Common selectors for send button (adjust if WhatsApp Web updates)
                            send_button = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, "span[data-testid='send']"))
                            )
                            send_button.click()
                            print("[WhatsApp Monitor] Clicked send button.")

                        except StaleElementReferenceException:
                            print("[WhatsApp Monitor] Stale element encountered, re-finding elements...")
                            # Re-find elements and try again (basic retry, more robust solutions might involve loops)
                            try:
                                input_field = WebDriverWait(driver, 10).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[contenteditable='true'][data-tab='10']"))
                                )
                                input_field.send_keys(ai_response_text)
                                send_button = WebDriverWait(driver, 10).until(
                                    EC.element_to_be_clickable((By.CSS_SELECTOR, "span[data-testid='send']"))
                                )
                                send_button.click()
                                print("[WhatsApp Monitor] Retried: Typed AI response and clicked send button.")
                            except Exception as retry_e:
                                print(f"[WhatsApp Monitor] Error during retry to send message: {retry_e}")
                        except NoSuchElementException:
                            print("[WhatsApp Monitor] Input field or send button not found. Cannot reply.")
                        except Exception as e_send:
                            print(f"[WhatsApp Monitor] Error sending reply via Selenium: {e_send}")
                        # --- End: Send AI Response via Selenium ---

                        # If original message was audio, or if all replies should be audio, convert to TTS
                        # For now, let's assume we reply in audio if the original message was audio.
                        # Sending actual audio files via Selenium involves more complex file upload or drag-and-drop simulations.
                        if message_type == "AUDIO":
                            ai_response_audio_path = text_to_speech(ai_response_text)
                            if ai_response_audio_path:
                                print(f"[WhatsApp Monitor] AI Response Audio saved to: {ai_response_audio_path} (but not sent via Selenium for simplicity).")
                            else:
                                print("[WhatsApp Monitor] Failed to generate AI response audio.")

                    else:
                        print(f"[WhatsApp Monitor] No valid user input text to process for message {message_id}.")

                    processed_message_ids.add(message_id)
                    print(f"[WhatsApp Monitor] Message {message_id} added to processed_message_ids.")

                except StaleElementReferenceException as e:
                    print(f"[WhatsApp Monitor] StaleElementReferenceException encountered while processing message. Re-fetching messages. Error: {e}")
                    # Break the inner loop to re-fetch the 'messages' list in the next outer loop iteration
                    break
                except Exception as e:
                    print(f"Error processing single message in monitor_whatsapp_messages loop: {e}")
                    continue # Continue to next message if one fails

        except Exception as e:
            # StaleElementReferenceException is common when page updates dynamically
            print(f"Error monitoring WhatsApp messages: {e}")
            # Optionally, re-initialize elements or driver if errors are persistent

        time.sleep(3) # Pause for 3 seconds before checking again