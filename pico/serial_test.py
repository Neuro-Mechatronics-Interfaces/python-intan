import time
from utilities.messaging_utilities import PicoMessager

def main():
    # Create an instance of PicoMessager
    pico_messenger = PicoMessager(port='/dev/ttyACM0', baudrate=9600, verbose=True)

    gestures = ["wave", "point", "fist", "open", "thumbs_up"]

    try:
        while True:
            # Randomly update gestures for demonstration
            current_gesture = gestures[int(time.time()) % len(gestures)]
            pico_messenger.update_gesture(current_gesture)

            # Check for messages from the Pico
            #pico_messenger.check_for_messages()
            pico_messenger.dump_output()

            # Wait a bit before the next iteration
            time.sleep(0.5)  # Adjust as needed for the rate of sending messages

    except KeyboardInterrupt:
        # Graceful exit on user interruption
        print("Stopping script.")
    finally:
        # Close serial connection
        pico_messenger.close_connection()

if __name__ == "__main__":
    main()
