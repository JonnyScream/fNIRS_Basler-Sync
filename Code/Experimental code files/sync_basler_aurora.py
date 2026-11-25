import time
import cv2
from pypylon import pylon
from pylsl import StreamInfo, StreamOutlet

# 1. Create the LSL Marker Stream for Aurora
info = StreamInfo(name='Trigger', type='Markers', channel_count=1, nominal_srate=0, channel_format='string', source_id='basler_cam_01')
outlet = StreamOutlet(info)

print("LSL Stream 'BaslerLaserTriggers' is active.")
print("ACTION REQUIRED: Go to Aurora -> Click the refresh icon next to 'Trigger stream' at the bottom -> Make sure the dot turns GREEN.")
input("\nPress ENTER here once Aurora's 'Trigger stream' dot is GREEN and Aurora is recording...")

# 2. Connect to the Basler Camera
# IMPORTANT: Close pylon Viewer before running this script, as only one app can access the camera at a time.
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Configure image converter to BGR for OpenCV video saving
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

width = camera.Width.Value
height = camera.Height.Value
fps = 30.0  # Adjust to match your desired recording frame rate

# 3. Setup Video Writer
video_filename = f"laser_recording_{int(time.time())}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

# 4. Start Recording & Send Start Trigger to Aurora
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
outlet.push_sample(['RECORDING_START'])
print(f"Sent 'RECORDING_START' trigger to Aurora. Recording video to {video_filename}...")
print("Press 'q' in the video preview window to stop recording.")

frame_count = 0
try:
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            out.write(img)
            frame_count += 1

            # Optional: Send a marker every 30 frames (approx. once per second) to verify ongoing sync
            if frame_count % 30 == 0:
                outlet.push_sample([f'FRAME_{frame_count}'])

            cv2.imshow('Basler Laser Feed (Press Q to stop)', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        grabResult.Release()
finally:
    # 5. Send Stop Trigger & Clean Up
    outlet.push_sample(['RECORDING_STOP'])
    print("Sent 'RECORDING_STOP' trigger to Aurora.")
    camera.StopGrabbing()
    camera.Close()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames successfully.")
