import os
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import io
import json
import base64
from gtts import gTTS

# Import for displaying QR code directly in Colab output
from IPython.display import Image, display

# Selenium imports
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import tempfile
import threading # Import threading
# Import all necessary components from .utils
from .utils import (
    stt_model, speech_to_text, text_to_speech, handle_uploaded_pdf,
    process_pdf_to_vectorstore, initialize_webdriver, monitor_whatsapp_messages,
    tokenizer, model, embeddings, FAISS_INDEX_DIR, get_rag_llm_response
)

# The functions and global variables are defined in the previously executed cell (6IIuLd9DcqUf)
# and are therefore globally available. The relative import is not needed here.
# from .utils import (
#     stt_model, speech_to_text, text_to_speech, handle_uploaded_pdf,
#     process_pdf_to_vectorstore, initialize_webdriver, monitor_whatsapp_messages,
#     tokenizer, model, embeddings, FAISS_INDEX_DIR, get_rag_llm_response
# )

# Global variable for Selenium WebDriver
whatsapp_driver = None
# Global variable for the monitoring thread
monitor_thread_instance = None

def home_view(request):
    return HttpResponse("Welcome to AI Agent Project! Use /llm_inference/, /upload_pdf/, or /whatsapp_webhook/ endpoints.")

print("HuggingFaceEmbeddings initialized for RAG. (from utils)")
print(f"FAISS_INDEX_DIR: {FAISS_INDEX_DIR} (from utils)")

def llm_inference_view(request):
    user_prompt = request.GET.get('prompt', "What is the capital of France?")

    try:
        # Use the refactored utility function for RAG and LLM inference
        response_text = get_rag_llm_response(user_prompt)

        print("Inference successful with GPT-2.")

        return JsonResponse(
            {
                "prompt": user_prompt,
                "response": response_text
            }
        )

    except Exception as e:
        print(f"Error during GPT-2 inference: {e}")
        return HttpResponse(f"Error during GPT-2 inference: {e}", status=500)

@csrf_exempt
def whatsapp_webhook_view(request):
    """
    Receive WhatsApp messages/calls:
    - Voice message -> STT -> GPT response -> TTS -> reply
    - Text message -> GPT response -> TTS optional
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"}, status=400)

    try:
        data = json.loads(request.body)
        message_type = data.get("type", "text")  # text or audio
        user_input = data.get("message", "")

        # 1. If audio, convert to text using stt_model from utils
        if message_type == "audio":
            audio_path = data.get("audio_path")  # WhatsApp audio saved locally
            if stt_model and os.path.exists(audio_path):
                # Use speech_to_text from utils
                user_input = speech_to_text(audio_path)
            else:
                print(f"STT model not loaded or audio file not found at {audio_path}. Cannot process audio.")
                user_input = ""

        # 2. RAG + GPT2 inference using the refactored utility function
        response_text = get_rag_llm_response(user_input)

        # 3. Convert GPT text to speech (TTS) using gTTS
        tts = gTTS(text=response_text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # 4. Return JSON with text + audio (base64 optional for WhatsApp)
        audio_base64 = base64.b64encode(mp3_fp.read()).decode("utf-8")

        return JsonResponse({
            "status": "success",
            "user_input": user_input,
            "response_text": response_text,
            "response_audio_base64": audio_base64
        })

    except Exception as e:
        print(f"Error in whatsapp_webhook_view: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt # For simplicity, disable CSRF for this view in a demo environment
def upload_pdf_view(request):
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        uploaded_file = request.FILES['pdf_file']
        print(f"Received PDF file: {uploaded_file.name}")

        # 1. Save the uploaded file
        file_path = handle_uploaded_pdf(uploaded_file)

        # 2. Process the PDF and store in vector store
        if file_path:
            # Use embeddings from utils
            index_path = process_pdf_to_vectorstore(file_path, embeddings)
            if index_path:
                return JsonResponse({"status": "success", "message": f"PDF file '{uploaded_file.name}' received and processed successfully.", "filename": uploaded_file.name, "vector_store_path": index_path})
            else:
                return JsonResponse({"status": "error", "message": f"Failed to process PDF file '{uploaded_file.name}' into vector store."}, status=500)
        else:
            return JsonResponse({"status": "error", "message": f"Failed to save PDF file '{uploaded_file.name}'"}, status=500)
    return JsonResponse({"status": "error", "message": "Please upload a PDF file via POST request."}, status=400)

@csrf_exempt
def is_driver_alive(driver):
    """
    Checks if the Selenium WebDriver is still running.
    Returns True if alive, False if not.
    """
    try:
        driver.title  # try accessing a property
        return True
    except:
        return False



@csrf_exempt
def whatsapp_session_view(request):
    """
    Handles WhatsApp Web session management (start, monitor, stop).
    """
    global whatsapp_driver, monitor_thread_instance

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            action = data.get('action')

            # =========================
            # 🚀 START SESSION
            # =========================
            if action == 'start':
                print("Attempting to start WhatsApp session...")

                # Check driver and ensure it's properly quit if it exists but is not alive
                if whatsapp_driver and not is_driver_alive(whatsapp_driver):
                    print("Existing driver found but not alive, attempting to quit...")
                    try:
                        whatsapp_driver.quit()
                    except Exception as e:
                        print(f"Error quitting old driver: {e}")
                    whatsapp_driver = None # Reset driver

                if whatsapp_driver is None:
                    print("Creating NEW Chrome session...")
                    # Unpack the tuple returned by initialize_webdriver
                    new_driver, logged_in_status = initialize_webdriver()
                    whatsapp_driver = new_driver # Assign the driver to the global variable
                    print("Selenium WebDriver initialized.")

                    if logged_in_status is False:
                        # QR code needed
                        return JsonResponse({"status": "qr_required", "message": "WhatsApp Web opened. Please scan the QR code from your phone to log in."})
                    elif logged_in_status is True:
                        # Already logged in
                        return JsonResponse({"status": "success", "message": "WhatsApp already logged in."})
                    else:
                        # Unknown status (error during init_webdriver)
                        return JsonResponse({"status": "error", "message": "Could not determine login status during WebDriver initialization."}, status=500)
                else:
                    # If whatsapp_driver is already active and alive from a previous call
                    return JsonResponse({"status": "success", "message": "WhatsApp session already active."})

            # =========================
            # 🦾 MONITOR MESSAGES
            # =========================
            elif action == 'monitor':
                if whatsapp_driver is None or not is_driver_alive(whatsapp_driver):
                    return JsonResponse({"status": "error", "message": "Driver not active. Start session first."}, status=400)

                # Ensure only one monitor thread runs at a time
                if monitor_thread_instance is None or not monitor_thread_instance.is_alive():
                    print("Starting new monitor thread...")
                    monitor_thread_instance = threading.Thread(target=monitor_whatsapp_messages, args=(whatsapp_driver,), daemon=True)
                    monitor_thread_instance.start()
                    return JsonResponse({"status": "success", "message": "Monitoring started."})
                else:
                    print("Monitor thread already running.")
                    return JsonResponse({"status": "success", "message": "Monitoring already active."})

            # =========================
            # 🛑 STOP SESSION
            # =========================
            elif action == 'stop':
                if whatsapp_driver:
                    try:
                        whatsapp_driver.quit()
                        whatsapp_driver = None
                        # Optionally stop the monitor thread if it's running
                        if monitor_thread_instance and monitor_thread_instance.is_alive():
                            # In a real application, you'd need a way to signal the thread to stop gracefully.
                            # For this example, we'll just acknowledge it's running and might eventually stop.
                            print("Monitor thread is still running, but driver is quit. It will eventually exit.")
                        monitor_thread_instance = None
                        return JsonResponse({"status": "success", "message": "WhatsApp session stopped."})
                    except Exception as e:
                        return JsonResponse({"status": "error", "message": f"Error stopping driver: {e}"}, status=500)
                else:
                    return JsonResponse({"status": "error", "message": "No active session to stop."}, status=400)

            # =========================
            # 🗲 STATUS CHECK
            # =========================
            elif action == 'status':
                driver_status = 'inactive'
                monitor_status = 'inactive'

                if whatsapp_driver and is_driver_alive(whatsapp_driver):
                    driver_status = 'active'
                    try:
                        current_url = whatsapp_driver.current_url
                        if 'web.whatsapp.com' in current_url:
                            driver_status = 'active and on WhatsApp Web'
                            # Check if logged in (pane-side element is usually present)
                            try:
                                WebDriverWait(whatsapp_driver, 5).until(EC.presence_of_element_located((By.ID, "pane-side")))
                                driver_status += ' (logged in)'
                            except:
                                driver_status += ' (QR code awaiting scan)'
                        else:
                            driver_status += f' (at {current_url})'
                    except Exception as e:
                        driver_status += f' (error checking URL: {e})'


                if monitor_thread_instance and monitor_thread_instance.is_alive():
                    monitor_status = 'active'

                return JsonResponse({
                    "status": "success",
                    "message": "Current session status",
                    "driver_status": driver_status,
                    "monitor_thread_status": monitor_status
                })

            else:
                return JsonResponse({"status": "error", "message": f"Unknown action '{action}'"}, status=400)

        except Exception as e:
            print(f"Error in whatsapp_session_view: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "POST request required"}, status=400)