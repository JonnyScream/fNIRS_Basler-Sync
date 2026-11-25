import time
import csv
import cv2
from pypylon import pylon
from pylsl import StreamInfo, StreamOutlet, local_clock

# 1. Create an LSL stream named 'Trigger' for Aurora
info = StreamInfo(name='Trigger', type='Markers', channel_count=1, nominal_srate=0, channel_format='string', source_id='basler_cam_01')
outlet = StreamOutlet(info)

input("1. Verify that 'Trigger stream' is green in Aurora.\n2. Start recording in Aurora.\n3. Press ENTER here to start face recording...")

# 2. Connect to the camera and configure output files
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

timestamp_tag = int(time.time())
video_filename = f"face_video_{timestamp_tag}.avi"
csv_filename = f"face_timestamps_{timestamp_tag}.csv"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 30.0, (camera.Width.Value, camera.Height.Value))

# 3. Start recording and send a single START trigger
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
outlet.push_sample(['START'])
print("Face recording started and START trigger sent. Press 'q' in the video window to stop.")

frame_idx = 0
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame_index', 'lsl_timestamp'])

    try:
        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                t_now = local_clock()
                img = converter.Convert(grabResult).GetArray()
                out.write(img)
                writer.writerow([frame_idx, t_now])
                frame_idx += 1

                cv2.imshow('Face Recording (Press Q to stop)', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            grabResult.Release()
    finally:
        # 4. Send a single STOP trigger and close resources
        outlet.push_sample(['STOP'])
        camera.StopGrabbing()
        camera.Close()
        out.release()
        cv2.destroyAllWindows()
        print(f"Finished! Saved {frame_idx} frames to {video_filename} and timestamps to {csv_filename}.")
