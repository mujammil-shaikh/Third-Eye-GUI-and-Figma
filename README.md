# Vision Voices

Vision Voices is a revolutionary multimodal AI solution designed to empower visually impaired individuals by bridging the gap between visual and auditory information. This project combines computer vision, speech recognition, natural language processing, and language translation to provide an inclusive and accessible experience.

## Features

- **Multimodal Interface**: Users can capture or upload images, and provide audio queries about the visual content.
- **Multilingual Support**: Speech-to-text and text-to-speech capabilities support multiple languages, enabling users from diverse linguistic backgrounds to access the application.
- **Tailored Image Descriptions**: The Large Language and Vision Assistance (LLaVA) model generates detailed, natural language descriptions of images, specifically tailored for visually impaired individuals based on their audio queries.
- **User-Friendly GUI**: The intuitive graphical user interface (GUI) simplifies the process of image capture, audio recording, query submission, and audio playback, making the technology accessible to users with varying levels of technical expertise.
- **Feedback Mechanisms**: Users can provide feedback on the generated descriptions by upvoting, downvoting, or flagging them, contributing to the continuous improvement of the system.

## Technologies Used

- **Programming Language**: Python
- **GUI Framework**: Tkinter
- **Image Processing**: OpenCV, PIL (Python Imaging Library)
- **Audio Processing**: sounddevice, soundfile, pydub
- **Speech-to-Text (STT)**: OpenAI Whisper (Large-v3) Model (via Hugging Face Inference API)
- **Text-to-Speech (TTS)**: Meta's mms-tts API (via Hugging Face Inference API)
- **Language Detection and Translation**: langdetect, deep_translator (Google Translate API)
- **Vision and Language Model**: LLaVA (Large Language and Vision Assistance) Model (via Gradio)
- **Third-Party Tools and Services**: Git, GitHub, Virtual Environment (e.g., venv, conda), IDE/Text Editor (e.g., PyCharm, Visual Studio Code)

## Getting Started

1. Clone the repository: `git clone https://github.com/mujammil-shaikh/Third-Eye-GUI-and-Figma.git`
2. Navigate to the project directory: `cd Third-Eye-GUI-and-Figma`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the application: `python STT_LLaVA_TTS_GUI_APP.py`

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [OpenAI Whisper](https://openai.com/blog/whisper/)
- [Meta's mms-tts API](https://huggingface.co/models?other=mms-tts)
- [LLaVA Model](https://github.com/haotian-liu/LLaVA)
- [Gradio](https://github.com/gradio-app/gradio)
- [OpenCV](https://opencv.org/)
- [PIL](https://python-pillow.org/)
- [sounddevice](https://python-sounddevice.readthedocs.io/en/0.4.1/)
- [soundfile](https://python-soundfile.readthedocs.io/en/0.10.3.post1/)
- [pydub](https://github.com/jiaaro/pydub)
- [langdetect](https://github.com/Mimino666/langdetect)
- [deep_translator](https://github.com/nidhaloff/deep_translator)
