import cv2
import pyaudio
import wave
import threading

def video_capture():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def audio_capture(filename="output.wav"):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True)
    frames = []

    for _ in range(0, int(44100 / 1024 * 5)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

def start_capture():
    video_thread = threading.Thread(target=video_capture)
    audio_thread = threading.Thread(target=audio_capture)

    video_thread.start()
    audio_thread.start()

    video_thread.join()
    audio_thread.join()

if __name__ == "__main__":
    start_capture()