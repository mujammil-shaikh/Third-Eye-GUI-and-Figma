import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import sys
from gradio_client import Client
import requests
from langdetect import detect
from deep_translator import GoogleTranslator
from datetime import datetime
from pydub import AudioSegment
import os
import io
import time
from requests.exceptions import HTTPError, RequestException

class ChatbotGUI(tk.Tk):
    IMAGE_SIZE = (300, 300)

    def __init__(self):
        super().__init__()
        self.title("Vision + Audio to Voice Chatbot")
        self.geometry("1500x800")
        self.fs = 90000  # Sample rate
        self.recording = np.array([])  # Array to hold audio data
        self.is_recording = False  # Flag to control recording
        self.q = queue.Queue()  # Queue to hold audio chunks
        self.image_filename = None  # Initialize filename variable
        self.audio_filename = None  # Initialize filename variable
        self.STT_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        self.TTS_API_BASE_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-"
        self.headers = {"Authorization": "Bearer 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"}
        self.client = Client("https://llava.hliu.cc/")
        self.photo_image = None
        self.ai_response = None
        self.user_lang = None
        self.user_lang_for_trans =  {
                                        'en': 'en', 'hi': 'hi', 'es': 'es',
                                        'fr': 'fr', 'ar': 'ar', 'bn': 'bn', 'ru': 'ru',
                                        'pt': 'pt', 'ur': 'ur', 'id': 'id',
                                        'de': 'de', 'ja': 'ja', 'ha': 'ha', 'ig': 'ig',
                                        'yo': 'yo', 'mr': 'mr', 'te': 'te', 'tr': 'tr',
                                        'ta': 'ta', 'vi': 'vi', 'gu': 'gu',
                                        'kn': 'kn', 'or': 'or', 'ml': 'ml'
                                    }
        self.user_query = None
        self.trans_user_query = None
        self.ai_response_audio = None
        self.playing = False
        self.bot_filename = "ai_response.wav"
        self.last_bot_response_index = "1.0" 
        # Language codes
        self.language_codes = {
            'en': 'eng', 'zh-cn': 'hak', 'hi': 'hin', 'es': 'spa',
            'fr': 'fra', 'ar': 'ara', 'bn': 'ben', 'ru': 'rus',
            'pt': 'por', 'ur': 'urd-script_arabic', 'id': 'ind',
            'de': 'deu', 'ja': 'jpn', 'ha': 'hau', 'ig': 'ibo',
            'yo': 'yor', 'mr': 'mar', 'te': 'tel', 'tr': 'tur',
            'ta': 'tam', 'vi': 'vie', 'gu': 'guj',
            'kn': 'kan', 'or': 'ori', 'ml': 'mal'
        }


         # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.show_camera_preview = False



        self.setup_ui()

    def setup_ui(self):
        self.create_menu()
        self.create_side_panel()
        self.create_main_section()

    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.on_close)

    def on_close(self):
        self.stop_camera()
        self.stop_recording()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.destroy()

    def show_camera(self):
        if hasattr(self, 'cap') and self.cap.isOpened() and self.show_camera_preview:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.IMAGE_SIZE)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.after(10, self.show_camera)

    def stop_camera(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.hide_camera_buttons()
            self.show_initial_buttons()
            self.show_camera_preview = False
            self.image_label.configure(image='')
            self.image_label.image = None

    def create_side_panel(self):
        side_panel = tk.Frame(self, bg="lightgray", width=300, height=800)
        side_panel.pack_propagate(False)
        side_panel.pack(side="left", fill="y")

        self.create_image_section(side_panel)
        self.create_audio_section(side_panel)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_main_section(self):
        self.main_section = tk.Frame(self)
        self.main_section.place(x=300, y=0, relwidth=0.8, relheight=1)

        self.title_label = tk.Label(self.main_section, text="Vision + Audio to Voice Chatbot", bg="lightgray", font=("Helvetica", 16))
        self.title_label.pack(fill="x", padx=10, pady=10)

        self.system_prompt_label = tk.Label(self.main_section, text="System Prompt", bg="lightgray")
        self.system_prompt_label.pack(fill="x", padx=10, pady=5)

        default_prompt_text = """Your task is to describe an image in a natural and human-like way. Imagine you are talking to a blind person who cannot see the image. 
Write in the second person, using “you” and “your”. Your description should be about 100 words long.

First, tell the blind person the most important information that they would want to know, such as the main subject, the background, and the mood of the image. 
For example, you could say “You are looking at a photo of a smiling woman holding a baby in a park.”

Then, add more details that make the image vivid and interesting, such as the colors, shapes, textures, and actions. 
For example, you could say “The woman has long brown hair and wears a blue dress. The baby is wearing a yellow hat and has chubby cheeks. 
They are sitting on a green blanket under a big tree. The sun is shining and there are some birds in the sky.”

Here is an example of a good description for a sample image:

“You are looking at a painting by Vincent van Gogh called The Starry Night. The sky is dark blue and filled with bright stars and swirls of yellow and white. 
There is a big moon on the top right corner. The town has a few houses and a church with a tall spire. The houses have red roofs and yellow windows. 
There is a river that reflects the light of the sky. 
There are some hills and trees in the background. The painting looks magical and peaceful.”"""


        self.system_prompt_entry = tk.Text(self.main_section , height=6)
        self.system_prompt_entry.insert("1.0", default_prompt_text)
        self.system_prompt_entry.pack(fill="x", padx=10, pady=5)

        self.chat_conversation_box = tk.Text(self.main_section, bg="white", height=18)
        self.chat_conversation_box.pack(fill="both", expand=True, padx=10, pady=5)

        self.feedback_section = tk.Frame(self.main_section, bg="lightgray")
        self.feedback_section.pack(fill="x", padx=10, pady=5)

        self.audio_player_feedback = tk.Frame(self.feedback_section, bg="white")
        self.audio_player_feedback.pack(fill="x", padx=10, pady=5)

        self.resend_tts_btn = tk.Button(self.audio_player_feedback, text="Resend For TTS", command=lambda: threading.Thread(target=self.resend_tts).start())
        self.resend_tts_btn.pack(side="left", padx=10, pady=5)

        self.play_ai_audio_btn = tk.Button(self.audio_player_feedback, text="Play Audio", command=lambda: threading.Thread(target=self.play_ai_audio).start())
        self.play_ai_audio_btn.pack(side="left", padx=10, pady=5)

        self.upvote_btn = tk.Button(self.audio_player_feedback, text="Upvote", command=lambda: threading.Thread(target=self.upvote).start())
        self.upvote_btn.pack(side="left", padx=10, pady=5)

        self.downvote_btn = tk.Button(self.audio_player_feedback, text="Downvote", command=lambda: threading.Thread(target=self.downvote).start())
        self.downvote_btn.pack(side="left", padx=10, pady=5)

        self.flag_btn = tk.Button(self.audio_player_feedback, text="Flag", command=lambda: threading.Thread(target=self.flag).start())
        self.flag_btn.pack(side="left", padx=10, pady=5)

        # Assign the text_input_field attribute
        self.text_input_field = tk.Entry(self.main_section)
        self.text_input_field.pack(fill="x", padx=10, pady=5)

        self.send_query_btn = tk.Button(self.main_section, text="Send Query", command=lambda: threading.Thread(target=self.send_query).start())
        self.send_query_btn.pack(fill="x", padx=10, pady=5)

        # Additional buttons
        self.clear_chat_btn = tk.Button(self.main_section, text="Clear Chat", command=lambda: threading.Thread(target=self.clear_chat).start())
        self.clear_chat_btn.pack(side="left", padx=10, pady=5)

        self.save_chat_btn = tk.Button(self.main_section, text="Save Chat", command=lambda: threading.Thread(target=self.save_chat).start())
        self.save_chat_btn.pack(side="left", padx=10, pady=5)

        self.regenerate_btn = tk.Button(self.main_section, text="Regenerate Previous Response", command=lambda: threading.Thread(target=self.regenerate_response).start())
        self.regenerate_btn.pack(side="left", padx=10, pady=5)

    def create_image_section(self, parent):
        self.image_section = tk.Frame(parent, bg="white", width=300, height=560)
        self.image_section.pack_propagate(False)
        self.image_section.pack(side="top", fill="y")

        self.image_label = tk.Label(self.image_section, bg="lightgray")
        self.image_label.pack(fill="both", expand=True)

        self.choose_image_btn = tk.Button(self.image_section, text="Choose Image", command=lambda: threading.Thread(target=self.choose_image).start())
        self.choose_image_btn.pack(side="top", padx=10, pady=5)

        self.use_camera_btn = tk.Button(self.image_section, text="Use Camera",command=lambda: threading.Thread(target=self.use_camera).start())
        self.use_camera_btn.pack(side="top", padx=10, pady=5)

        self.close_preview_btn = tk.Button(self.image_section, text="Close Preview", command=lambda: threading.Thread(target=self.close_preview).start())
        self.close_preview_btn.pack(side="top", padx=10, pady=5)

        self.capture_image_btn = tk.Button(self.image_section, text="Capture Image", command=lambda: threading.Thread(target=self.capture_image).start())
        self.capture_image_btn.pack(side="top", padx=10, pady=5)

        self.hide_camera_buttons()

    def create_audio_section(self, parent):
        self.audio_section = tk.Frame(parent, bg="white", width=300, height=240)
        self.audio_section.pack_propagate(False)
        self.audio_section.pack(side="top", fill="y", pady=(10, 0))

        self.audio_section = tk.Frame(self.audio_section, width=300, height=560, bg='white')
        self.audio_section.pack(side=tk.BOTTOM, fill=tk.BOTH)

        self.file_name_label = tk.Label(self.audio_section, text=f"-----{self.audio_filename}-----")
        self.file_name_label.pack()

        self.create_audio_buttons()

    def create_audio_buttons(self):
        self.play_button = tk.Button(self.audio_section, text="Play", state=tk.NORMAL, command=lambda: threading.Thread(target=self.play_audio).start())
        self.play_button.pack()

        self.stop_button = tk.Button(self.audio_section, text="Stop", state=tk.DISABLED, command=lambda: threading.Thread(target=self.stop_audio).start())
        self.stop_button.pack()

        # Create the start and stop buttons with initial states and texts
        self.start_rec_button = tk.Button(self.audio_section, text="Start Recording", state=tk.NORMAL, command=lambda: threading.Thread(target=self.start_recording).start())
        self.start_rec_button.pack()

        self.stop_rec_button = tk.Button(self.audio_section, text="Stop Recording", state=tk.DISABLED, command=lambda: threading.Thread(target=self.stop_recording).start())
        self.stop_rec_button.pack()

        self.choose_audio_button = tk.Button(self.audio_section, text="Choose Audio File", command=lambda: threading.Thread(target=self.choose_audio).start())
        self.choose_audio_button.pack()

        self.send_audio_button = tk.Button(self.audio_section, text="Send Audio", command=lambda: threading.Thread(target=self.send_audio).start())
        self.send_audio_button.pack()

    def show_initial_buttons(self):
        self.choose_image_btn.pack(side="top", padx=10, pady=5)
        self.use_camera_btn.pack(side="top", padx=10, pady=5)

    def hide_camera_buttons(self):
        self.close_preview_btn.pack_forget()
        self.capture_image_btn.pack_forget()

    def show_camera_buttons(self):
        self.close_preview_btn.pack(side="top", padx=10, pady=5)
        self.capture_image_btn.pack(side="top", padx=10, pady=5)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        self.image_filename = file_path
        if file_path:
            self.hide_camera_buttons()
            self.show_image(file_path)

    def use_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.hide_camera_buttons()
        self.show_camera_buttons()
        self.show_camera_preview = True
        self.show_camera()



    def show_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize(self.IMAGE_SIZE, Image.BILINEAR)
        photo = ImageTk.PhotoImage(image=image)
        self.photo_image = photo  # Store the PhotoImage in the attribute
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def show_image_from_camera(self, frame):
        frame = cv2.resize(frame, self.IMAGE_SIZE)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.photo_image = photo  # Store the PhotoImage in the attribute
        self.image_label.configure(image=photo)
        self.image_label.image = photo


    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_image_from_camera(frame)
            self.hide_camera_buttons()
            self.show_initial_buttons()
            self.show_camera_preview = False
            self.save_image_from_camera(frame)  # Save the captured image
            self.release_camera()  # Release the camera here

    def close_preview(self):
        self.hide_camera_buttons()
        self.show_initial_buttons()
        self.show_camera_preview = False
        self.image_label.configure(image='')
        self.image_label.image = None
        self.release_camera()  # Release the camera here as well

    def release_camera(self):
        if self.cap:
            self.cap.release()


    def save_image_from_camera(self, frame):
        filename = f"captured_image.png"
        self.image_filename = filename
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Image saved as {filename}")

    def play_audio(self):
        # Disable the start button and enable the stop button
        self.play_button.config(state=tk.DISABLED, text="Playing...")
        self.stop_button.config(state=tk.NORMAL, text="Stop Playing")
        if self.audio_filename:
            self.data, self.fs = sf.read(self.audio_filename, dtype='float32')
            sd.play(self.data, self.fs)
              # Wait for the playback to finish
            sd.wait()
            # Update the text of the play button to 'Replay'
            self.play_button.after(800, self.update_buttons)  # 2000 milliseconds = 2 seconds
        else:
            print("No file selected")

    def update_buttons(self):
        # Update the text of the play button to 'Replay'
        self.play_button.config(state=tk.NORMAL, text="Replay")
        self.stop_button.config(state=tk.DISABLED, text="Stop")

    def stop_audio(self):
        # Disable the stop button and enable the start button
        self.stop_button.config(state=tk.DISABLED, text="Stopped Playing")
        self.play_button.config(state=tk.NORMAL, text="Start Playing")
        sd.stop()

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def start_recording(self):
        self.delete_existing_audio_file()
        # Disable the start button and enable the stop button
        self.start_rec_button.config(state=tk.DISABLED, text="Recording...")
        self.stop_rec_button.config(state=tk.NORMAL, text="Stop Recording")
        self.is_recording = True
        self.stream = sd.InputStream(callback=self.callback)
        self.stream.start()

    def delete_existing_audio_file(self):
        if os.path.exists('output.wav'):
            os.remove('output.wav')
            print(f"Deleted existing file: {'output.wav'}")
        else:
            print(f"No existing file found with the name: {'output.wav'}")

    def stop_recording(self):
        # Disable the stop button and enable the start button
        self.stop_rec_button.config(state=tk.DISABLED, text="Stopped")
        self.start_rec_button.config(state=tk.NORMAL, text="Start Recording")
        
        if hasattr(self, 'stream') and self.stream:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            self.audio_filename = 'output.wav'
            
            # Clear the previous recording
            self.recording = np.array([], dtype='float32')
            
            while not self.q.empty():
                self.recording = np.append(self.recording, self.q.get())
            
            sf.write(self.audio_filename, self.recording, self.fs)
            self.update_file_name_label()



    def update_file_name_label(self):
        file_name = os.path.basename(self.audio_filename)
        self.file_name_label.config(text=f"-----{file_name}-----")
        self.send_audio_button.config(text='Send Audio')

    def choose_audio(self):
        audio_filetypes = [("Audio Files", "*.mp3;*.wav;*.ogg")]
        self.audio_filename = filedialog.askopenfilename(filetypes=audio_filetypes)
        if self.audio_filename:
            print(f"Selected audio file: {self.audio_filename}")
        else:
            print("No audio file selected.")
        self.update_file_name_label()
        

    def send_audio(self):
        if hasattr(self, 'audio_filename') and self.audio_filename:
            self.send_audio_button.config(text='Sended')

            # Retry convert_audio_to_text function
            text_output = self.convert_audio_to_text()

            self.after(0, self.update_text_input, text_output)
            self.send_audio_button.config(text='Done')
        else:
            print("No audio file specified.")
            

    def update_text_input(self, text_output):
        try:
            # Check if text_input_field is not None before updating
            if self.text_input_field:
                self.text_input_field.delete("0", tk.END)  # Clear existing text in the entry field
                if text_output is not None:
                    self.text_input_field.insert("0", text_output)  # Insert new text at the beginning
                self.user_query = text_output
            else:
                print("Text input field is not initialized.")
        except Exception as e:
            print("Error updating text input field:", e)






    def convert_audio_to_text(self):
        audio_filename = self.audio_filename 
        try:
            with open(audio_filename, "rb") as f:
                audio_data = f.read()

            response = requests.post(self.STT_API_URL, headers=self.headers, data=audio_data)

            if response.status_code == 200 and 'application/json' in response.headers.get('content-type', ''):
                response_data = response.json()

                if 'error' in response_data:
                    error_message = response_data['error']
                    print(f"Error from API: {error_message}")
                    return None  # Handle the error appropriately in your application
                else:
                    text_output = response_data.get('text', '')
                    print(text_output)
                    return text_output
            else:
                print(f"Unexpected response: {response.status_code}, {response.text}")
                return None  # Handle unexpected response status or content type

        except Exception as e:
            print(f"Error during audio-to-text conversion: {e}")
            return None  # Handle unexpected errors during the conversion process


    def get_user_language(self):
        try:
            print(self.user_query)
            lang = detect(self.user_query)
            return lang
        except:
            return None

    def send_query(self):
        # Assuming text_output is the name of your widget displaying text
        self.user_query = self.text_input_field.get().strip()
        self.user_lang = self.get_user_language()
        print(f"User Language: {self.user_lang}")

        if self.user_lang:
            try:
                # Translate from detected language to English
                self.trans_user_query = GoogleTranslator(source=self.user_lang_for_trans[self.user_lang], target='en').translate(self.user_query)
                print(f"Translation: {self.trans_user_query}")
            except Exception as translation_error:
                print(f"Translation Error: {translation_error}")
                # Handle translation error here

        self.send_query_btn.config(text='Sended to LLaVA')
        time.sleep(1)  # Add a delay of 1 second
        # Retry send_query_ai function
        self.retry_function(self.send_query_ai, max_retries=5)
        self.send_query_btn.config(text='Sended to TTS')
        try:
            time.sleep(1)  # Add a delay of 1 second
            # Convert text to speech with retry
            self.retry_function(self.convert_text_to_speech, max_retries=5)
            self.send_query_btn.config(text='Done')
            time.sleep(1)
            self.send_query_btn.config(text='Send Query')
            time.sleep(1)
            if os.path.exists(self.bot_filename):
                self.play_ai_audio()
        except Exception as tts_error:
            print(f"TTS Error: {tts_error}")
            # Handle TTS error here
            self.send_query_btn.config(text='click Resend For TTS')
            time.sleep(1)
            self.send_query_btn.config(text='Send Query')

        




    def send_query_ai(self):
        try:
            # Send user query to chatbot
            user_query_prompt = self.trans_user_query
            system_prompt_text = self.system_prompt_entry.get("1.0", tk.END).strip()

            combine_prompt = "Main Prompt: " + system_prompt_text +"\n ---------- \n" + "\n" + "Blind person query: " + user_query_prompt
            
            # Convert PhotoImage to base64-encoded string
            imagepath = self.image_filename

            # Make predictions using /add_text endpoint
            print(f"Combine Prompt: {combine_prompt}")
            print(f"Image : {imagepath}")

            # Display user input 
            self.chat_conversation_box.insert(tk.END, "User: " + self.user_query + "\n")


            result_add_text = self.client.predict(combine_prompt, imagepath, "Crop", api_name="/add_text")

            # Make predictions using /http_bot endpoint
            result_http_bot = self.client.predict("llava-v1.6-34b", 0.5, 0.5, 200, api_name="/http_bot")

            # Check if the result is not None before accessing content
            if result_http_bot is not None:
                result_http_bot_content = result_http_bot[0][1]
                print(f"Prediction Result Content (http_bot): {result_http_bot_content}")

            # Translate the response
            self.ai_response = GoogleTranslator(source='en', target=self.user_lang_for_trans[self.user_lang]).translate(result_http_bot_content)

            # Display chatbot response
            
            self.chat_conversation_box.insert(tk.END, "Chatbot: " + self.ai_response + "\n")
            self.chat_conversation_box.insert(tk.END,"\n" + "\n" + "--------------------" +  "\n" + "\n")

        except Exception as e:
            # Handle exceptions and display error details
            print(f"Error occurred: {str(e)}")
            self.chat_conversation_box.insert(tk.END, "Error: " + str(e) + "\n")
            self.chat_conversation_box.insert(tk.END, "\n" + "\n" + "--------------------" +  "\n" + "\n")

        

    def delete_existing_file(self):
            if os.path.exists(self.bot_filename):
                os.remove(self.bot_filename)
                print(f"Deleted existing file: {self.bot_filename}")
            else:
                print(f"No existing file found with the name: {self.bot_filename}")

    def resend_tts(self):
            # Call the function to delete an existing file
            self.delete_existing_file()

            # Retry convert_text_to_speech function
            self.retry_function(self.convert_text_to_speech, max_retries=5)

    def retry_function(self, func, max_retries=5, delay_seconds=1):
        for attempt in range(max_retries):
            try:
                func()
                break  # Break out of the loop if successful
            except HTTPError as http_error:
                if http_error.response.status_code in [500, 502, 503, 408, 429]:
                    # Retry for specific HTTP error codes
                    print(f"Retrying due to HTTP error: {http_error}")
                    time.sleep(delay_seconds * (2 ** attempt))  # Exponential backoff
                else:
                    # Handle other HTTP errors
                    print(f"HTTP Error: {http_error}")
                    # Add additional handling as needed
                    break
            except RequestException as request_exception:
                # Handle connection errors
                print(f"Request Exception: {request_exception}")
                time.sleep(delay_seconds * (2 ** attempt))  # Exponential backoff
            except Exception as e:
                # Handle other exceptions
                print(f"An error occurred: {e}")
                break

    def convert_text_to_speech(self):
        self.delete_existing_file()
        self.play_ai_audio_btn.config(text='Sended')
        API_URL = f"{self.TTS_API_BASE_URL}{self.language_codes[self.user_lang]}"
        trans_AI_query = self.ai_response

        try:
            response = requests.post(API_URL, headers=self.headers, json={"inputs": trans_AI_query})
            response.raise_for_status()  # This will raise an exception if the status code is not 200
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print ("Something went wrong with the request",err)

        audio = AudioSegment.from_file(io.BytesIO(response.content), format="flac")
        audio.export(self.bot_filename, format="wav")
        print(f"Audio saved to {self.bot_filename}")
        self.play_ai_audio_btn.config(text='Ready')




    def play_ai_audio(self):
        # Play last bot response audio
        if self.bot_filename:
            try:
                if not self.playing:
                    # Read the .wav file
                    self.data, self.fs = sf.read(self.bot_filename, dtype='float32')  
                    sd.play(self.data, self.fs)
                    self.playing = True
                    self.play_ai_audio_btn.config(text='Stop Audio')  # Update button text
                    sd.wait()  # Wait for playback to finish
                    self.playing = False
                    self.play_ai_audio_btn.config(text='Play Audio')  # Update button text
                else:
                    sd.stop()  # Stop playback
                    self.playing = False
                    self.play_ai_audio_btn.config(text='Play Audio')  # Update button text

            except Exception as e:
                print(f"Error during audio playback: {str(e)}")
        else:
            print("No file selected")



    def upvote(self):
        # Upvote last bot response
        try:
            result_upvote = self.client.predict("llava-v1.6-34b", api_name="/upvote_last_response")
            print("Upvote response:", result_upvote)
        except Exception as e:
            print("Error:", e)

    def downvote(self):
        # Downvote last bot response
        try:
            result_downvote = self.client.predict("llava-v1.6-34b", api_name="/downvote_last_response")
            print("Downvote response:", result_downvote)
        except Exception as e:
            print("Error:", e)

    def flag(self):
        # Flag last bot response
        try:
            result_flag = self.client.predict("llava-v1.6-34b", api_name="/flag_last_response")
            print("Flag response:", result_flag)
        except Exception as e:
            print("Error:", e)

    def clear_chat(self):
        # Clear chat history
        try:
            result_clear_history = self.client.predict(api_name="/clear_history")
            self.chat_conversation_box.delete(1.0, tk.END)
        except Exception as e:
            self.chat_conversation_box.insert(tk.END, "Error: " + str(e) + "\n")

    def save_chat(self):

        time.sleep(0.1)

        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d___%H_%M_%S")

        # Get the text from the chat_conversation_box
        chat_text = self.chat_conversation_box.get("1.0", "end-1c")

        # Create a filename with the timestamp
        filename = f"chat_{timestamp}.txt"

        # Save the text to the file
    # Save the text to the file
        with open(filename, "w", encoding='utf-8') as file:
            file.write(chat_text)



 # Start at the beginning of the text widget

    def regenerate_response(self):
        # Get all the text in the chat conversation box
        all_text = self.chat_conversation_box.get("1.0", tk.END).strip()

        # Find the last occurrence of the delimiter
        delimiter = "\n\n--------------------\n\n"
        last_delimiter_index = all_text.rfind(delimiter)

        if all_text.count(delimiter) > 1:
            new_text = all_text[:last_delimiter_index] + delimiter
            self.chat_conversation_box.delete("1.0", tk.END)
            self.chat_conversation_box.insert(tk.END, new_text)
        else:
            self.chat_conversation_box.delete("1.0", tk.END)
        # Set the new text in the chat conversation box

        self.send_query_btn.config(text='Sended to LLaVA')
        time.sleep(1)  # Add a delay of 1 second
        # Retry send_query_ai function
        self.retry_function(self.send_query_ai, max_retries=5)
        self.send_query_btn.config(text='Sended to TTS')
        try:
            time.sleep(1)  # Add a delay of 1 second
            # Convert text to speech with retry
            self.retry_function(self.convert_text_to_speech, max_retries=5)
            self.send_query_btn.config(text='Done')
            time.sleep(1)
            self.send_query_btn.config(text='Send Query')
            time.sleep(1)
            if os.path.exists(self.bot_filename):
                self.play_ai_audio()
        except Exception as tts_error:
            print(f"TTS Error: {tts_error}")
            # Handle TTS error here
            self.send_query_btn.config(text='click Resend For TTS')
            time.sleep(1)
            self.send_query_btn.config(text='Send Query')
        










if __name__ == "__main__":
    app = ChatbotGUI()
    app.mainloop()
