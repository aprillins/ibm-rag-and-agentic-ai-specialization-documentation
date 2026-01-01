from gtts import gTTS
tts = gTTS("Halo, aku mau tanya. Andrew Darwis itu siapa sih? Dia itu sukanya main apa ya?", lang='id')
tts.save("simple_gtts.mp3")