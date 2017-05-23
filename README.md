# DiscoGAN-Tensorflow

Generate Spectrogram & Save Spectrogram into tf.recordfile

There are four different data.py files
1. data_speech.py
2. data_speech_LogarithmScale.py
3. data_synth.py
4. data_synth_LogarithmScale.py


If you want to generate speech spectrogram tf.recordfile
put your VCTK male speech files into "Male" directory
also
put your VCTK female speech files into "Female" directory.


If you want to generate synth spectrogram tf.recordfile
put your bass wav files into "bass" directory.
also
put your keyboard wav files into "keyboard" directory.

Each .py file generates spectrogram from the directories above and change it into one spectrogram tf.recordfile in following directories.

(when running data_synth.py)
"spectrograms_bass"
"spectrograms_keyboard"

(when running data_synth_LogarithmScale.py)
"log_spectrograms_bass"
"log_spectrograms_keyboard"

Same form is also applied to data_speech.py and data_speech_LogarithmScale.py




