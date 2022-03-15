from transformers import MarianMTModel, MarianTokenizer

class back_translation:
    def __init__(self, target_model_name='Helsinki-NLP/opus-mt-en-ROMANCE',
                 target_model_name_three_translation='Helsinki-NLP/opus-mt-fr-es',
                 en_model_name='Helsinki-NLP/opus-mt-ROMANCE-en'):

        self.target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
        self.target_model = MarianMTModel.from_pretrained(target_model_name)

        self.target_tokenizer_three_translation = MarianTokenizer.from_pretrained(target_model_name_three_translation)
        self.target_model_three_translation = MarianMTModel.from_pretrained(target_model_name_three_translation)

        self.en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
        self.en_model = MarianMTModel.from_pretrained(en_model_name)

        return

    def translate(self, text, three_translation,language="fr"):
        # Prepare the text data into appropriate format for the model
        if not three_translation:
            template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
            src_texts = template(text)
        else:
            src_texts = text
        # Tokenize the texts

        if language != "en" and not three_translation:
            # sample_text = text
            # encoded = self.target_tokenizer.prepare_seq2seq_batch(src_texts=[src_texts], return_tensors="pt")
            encoded = self.target_tokenizer([src_texts], return_tensors="pt")
            # Generate translation using model
            translated = self.target_model.generate(**encoded)
            # Convert the generated tokens indices back into text
            translated_texts = self.target_tokenizer.batch_decode(translated, skip_special_tokens=True)
        elif language != "en" and three_translation:
            encoded = self.target_tokenizer_three_translation([src_texts], return_tensors="pt")


            # Generate translation using model
            translated = self.target_model_three_translation.generate(**encoded)
            # Convert the generated tokens indices back into text
            translated_texts = self.target_tokenizer_three_translation.batch_decode(translated,
                                                                                    skip_special_tokens=True)
        else:
            encoded = self.en_tokenizer([src_texts], return_tensors="pt")


            # Generate translation using model
            translated = self.en_model.generate(**encoded)
            # Convert the generated tokens indices back into text
            translated_texts = self.en_tokenizer.batch_decode(translated, skip_special_tokens=True)

        return translated_texts

    def back_translation(self, text, source_lang="en", target_lang="fr", three_translation=False):
        # Translate from source to target language
        text_temp = self.translate(text, three_translation, language=target_lang)
        # Translate from target language back to source language
        back_translated_texts = self.translate(text_temp[0], three_translation, language=source_lang)

        return back_translated_texts

    def get_translate_example(self, text):
        # NOTE translation from english-france-english
        back_translation_text = self.back_translation(text, source_lang="en", target_lang="fr")
        back_translation_text = back_translation_text[0].replace(", ", " , ").replace(".", " . ").strip()
        back_translation_text = back_translation_text.split(' ')
        while '' in back_translation_text:
            back_translation_text.remove('')
        back_translation_text = " ".join(back_translation_text)

        # NOTE when they are the same we do 3 way translation  english-france-spanish-english
        if back_translation_text.lower() == text.lower():
            aug1_texts = self.translate(text, three_translation=False, language="fr")
            aug2_texts = self.translate(aug1_texts[0], three_translation=True, language="es")
            back_translation_text = self.translate(aug2_texts[0], three_translation=False, language="en")
            back_translation_text = back_translation_text[0].replace(", ", " , ").replace(".", " . ").strip()
            back_translation_text = back_translation_text.split(' ')
            while '' in back_translation_text:
                back_translation_text.remove('')
            back_translation_text = " ".join(back_translation_text)

        return back_translation_text
