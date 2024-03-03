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
            self.send_query_ai()
            self.send_query_btn.config(text='Sended to TTS')
            try:
                time.sleep(1)  # Add a delay of 1 second
                self.convert_text_to_speech()
            except Exception as tts_error:
                print(f"TTS Error: {tts_error}")
                # Handle TTS error here
                self.send_query_btn.config(text='click Resend For TTS')

            self.send_query_btn.config(text='Done')