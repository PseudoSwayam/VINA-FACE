import subprocess
import threading
import time

# Simple mechanism to prevent spamming TTS if multiple quick recognitions for same person
# This is a global lock for any speech. For per-person cooldown, use main.py logic.
tts_lock = threading.Lock()
last_tts_call_time = 0
TTS_MIN_INTERVAL = 0.5  # Minimum interval between any two 'say' calls to prevent overlap issues

def speak(text):
    """
    Speaks the given text using macOS 'say' command in a non-blocking thread.
    Ensures 'say' commands don't excessively overlap.
    """
    global last_tts_call_time

    def _speak_task():
        global last_tts_call_time
        with tts_lock: # Ensure only one thread modifies last_tts_call_time or calls 'say' at once
            current_time = time.time()
            if current_time - last_tts_call_time < TTS_MIN_INTERVAL:
                # If called too quickly, could wait, but for now, let's just skip if too close.
                # 'say' itself queues, so this is more about not flooding the 'say' process spawner.
                # print(f"TTS call too soon for: {text}. Skipping.") # Optional debug
                return 
            
            try:
                # print(f"TTS: {text}") # For debugging
                subprocess.run(['say', text], check=True)
                last_tts_call_time = time.time() # Update time after successful call
            except FileNotFoundError:
                print("Error: 'say' command not found. Ensure you are on macOS.")
            except subprocess.CalledProcessError as e:
                print(f"Error during 'say' command execution: {e}")
            except Exception as e:
                print(f"An unexpected error occurred in TTS: {e}")

    # Run in a separate thread to avoid blocking the main (OpenCV) loop
    thread = threading.Thread(target=_speak_task)
    thread.daemon = True  # Allows main program to exit even if TTS thread is active
    thread.start()