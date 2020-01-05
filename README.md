# Backend

Code auf Basis von https://github.com/davidsandberg/facenet

* Mit Backend/getphoto.py wurden von jeder Person Gesichter aufgenommen und in Backend/output/person abgelegt.
* Backend/train_tripletloss.py trainiert die SVM auf die Gesichter
* Backend/realtime_face_and_emotion_recognition.py startet die Face Detection, Recognition und Emotion Detection

# Frontend

Code auf Basis von https://magicmirror.builders

* Verarbeitung der ZeroMQ-Nachrichten ist im MagicMirror/modules/compliments/node_helper.js
* Interaktion mit anderen Modulen ist in MagicMirror/modules/compliments/compliments.js implementiert.
* Beispielnachrichten können mit MagicMirror/modules/compliments/send_example.py verschickt werden.
* zum Start im MagicMirror-Verzeichnis "npm install && npm start" ausführen.

