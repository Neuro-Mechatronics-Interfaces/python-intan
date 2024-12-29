import time
import argparse
from utilities.messaging_utilities import PicoMessager

if __name__ == "__main__":

    args = argparse.ArgumentParser(description="PicoMessager test script")
    args.add_argument("--port", type=str, default="/dev/ttyACM0", help="Port for the Pico")
    args.add_argument("--baudrate", type=int, default=9600, help="Baudrate for the Pico")
    args.add_argument("--verbose", type=bool, default=True, help="Verbose output")
    args = args.parse_args()


    gestures = ["wave", "point", "fist", "open", "thumbs_up"]

    # Create an instance of PicoMessager
    pico_messenger = PicoMessager(port=args.port, baudrate=args.baudrate, verbose=args.verbose)

    # Run the main function
    try:
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