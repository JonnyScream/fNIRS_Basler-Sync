import time
import serial

PORT = "COM5"
BAUDRATES = [9600, 115200, 57600, 19200, 38400]

print(f"מתחיל בדיקת פרוטוקולים וקצבי תקשורת מול {PORT}...")
print("הסתכלו על המסך של Aurora (ודאו שהגרפים רצים!)\n")

for baud in BAUDRATES:
    try:
        print(f"--- בודק קצב {baud} ---")
        with serial.Serial(port=PORT, baudrate=baud, timeout=1) as ser:
            time.sleep(0.5)
            
            # שיטה 1: פולס בינארי קלאסי (BrainProducts / NIRx TriggerBox)
            ser.write(bytes([0]))
            time.sleep(0.05)
            ser.write(bytes([1]))
            time.sleep(0.1)
            ser.write(bytes([0]))
            time.sleep(1.0)
            
            # שיטה 2: שליחת תו ASCII (נפוץ במתאמי USB-TTL פשוטים / Arduino)
            ser.write(b'1')
            time.sleep(1.0)

    except serial.SerialException as e:
        print(f"שגיאה בקצב {baud}: {e}")

print("\nהבדיקה הסתיימה! האם הופיעו קווים אנכיים ב-Aurora?")
