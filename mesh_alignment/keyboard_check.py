import keyboard

def print_pressed_keys(e):
    print(f"Key: {e.name}, Scan code: {e.scan_code}")

# Hook to all keyboard events
keyboard.hook(print_pressed_keys)
keyboard.wait('esc')  # Use the 'esc' key as a trigger to stop the program
