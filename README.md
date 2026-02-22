# ESP32 Monitor Mirror

Stream your computer screen to an Elecrow CrowPanel ESP32-S3 7.0" display over USB Serial. Touch the display to control your PC with mouse clicks.

## Hardware

- **Display**: [Elecrow CrowPanel ESP32-S3 7.0"](https://www.elecrow.com/esp32-display-7-inch-hmi-display-rgb-tft-lcd-touch-screen-support-lvgl.html) (800x480, RGB parallel, capacitive touch)
- **Chip**: ESP32-S3-WROOM-1-N4R8 (4MB Flash, 8MB PSRAM)
- **Display driver**: EK9716BD3, 16-bit RGB parallel bus
- **Touch controller**: GT911 (I2C)
- **Connection**: USB-C (Serial at 2Mbps)
- **Computer**: Windows PC with Python 3.7+

## Features

- Real-time screen mirroring at ~10 FPS
- Differential frame updates (only changed pixels are sent)
- Run-length encoded protocol for efficient bandwidth usage
- ACK-based flow control for reliable transmission
- Touch-to-mouse input: tap/drag on the display to control your PC
- Configurable rotation, threshold, and FPS

## Setup

### ESP32 (Arduino IDE)

1. Install [Arduino IDE](https://www.arduino.cc/en/software)
2. Add ESP32 board support (Preferences > Additional Board Manager URLs):
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Install boards: **Board Manager > esp32 by Espressif Systems**
4. Install libraries via **Library Manager**:
   - **LovyanGFX** (display driver for RGB parallel bus)
   - **TAMC_GT911** (touch controller)
5. Open `receiver/receiver.ino`
6. Board settings:

   | Setting | Value |
   |---------|-------|
   | Board | ESP32S3 Dev Module |
   | Flash Size | 4MB |
   | Partition Scheme | Huge APP (3MB No OTA / 1MB SPIFFS) |
   | PSRAM | OPI PSRAM |
   | USB CDC On Boot | **Enabled** |

7. Upload

### PC (Python)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pyserial
   ```
2. Connect the ESP32 via USB-C
3. Find the COM port (Device Manager > Ports)
4. Run:
   ```bash
   python transmitter.py --port COM13 --target-fps 10
   ```

## Usage

```bash
# Basic usage
python3 transmitter.py --port COM13

# Specific monitor (1-based index)
python3 transmitter.py --port COM13 --monitor-index 2

# Use largest monitor
python3 transmitter.py --port COM13 --prefer-largest

# Adjust FPS and sensitivity
python3 transmitter.py --port COM13 --target-fps 15 --threshold 8

# Rotate capture 90 degrees
python3 transmitter.py --port COM13 --rotate 90

# Send full frames (no diffing)
python3 transmitter.py --port COM13 --full-frame

# Capture only a specific region of the monitor
python3 transmitter.py --port COM13 --crop-x 100 --crop-y 50 --crop-width 800 --crop-height 480
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | *(required)* | Serial port (e.g. COM13, /dev/ttyACM0) |
| `--baud` | 2000000 | Baud rate |
| `--monitor-index` | leftmost | Monitor index (1-based) |
| `--prefer-largest` | off | Use largest monitor |
| `--target-fps` | 10 | Target frame rate |
| `--threshold` | 5 | Pixel change threshold (0-255) |
| `--full-frame` | off | Send every pixel every frame |
| `--max-updates-per-frame` | 2000 | Max runs per packet |
| `--rotate` | 0 | Rotation (0, 90, 180, 270) |
| `--crop-x` | 0 | Crop region X offset (pixels from monitor left) |
| `--crop-y` | 0 | Crop region Y offset (pixels from monitor top) |
| `--crop-width` | *(full)* | Crop region width (required for cropping) |
| `--crop-height` | *(full)* | Crop region height (required for cropping) |

## Protocol

Communication uses USB Serial at 2Mbps with a custom binary protocol.

### Display packets (PC -> ESP32)

```
Header: 'P' 'X' 'U' 'R' (4B) + version (1B, 0x02) + frame_id (uint32 LE) + count (uint16 LE)
Body:   count entries of: y (uint16 LE) + x0 (uint16 LE) + length (uint16 LE) + color (uint16 LE) = 8B each
```

The receiver sends a single ACK byte (`0x06`) after processing each packet.

### Touch packets (ESP32 -> PC)

```
'T' 'C' 'H' + x (uint16 LE) + y (uint16 LE) + type (uint8) = 8 bytes
type: 0 = press, 1 = move, 2 = release
```

Touch coordinates are mapped from display space (800x480) to the captured monitor's pixel coordinates and translated into Windows mouse events.

## Architecture

```
transmitter.py (PC)          USB Serial 2Mbps          receiver.ino (ESP32)
+-----------------+          +-----------+          +------------------+
| Screen capture  |  PXUR -> |           | -> PXUR | Parse & draw     |
| Diff + RLE      |          |  Serial   |         | LovyanGFX RGB    |
| Touch -> Mouse  |  <- TCH  |           |  TCH <- | GT911 touch read |
+-----------------+          +-----------+          +------------------+
```

## Troubleshooting

### No image on display
- Verify **USB CDC On Boot: Enabled** in Arduino IDE
- Check COM port in Device Manager
- Ensure the correct baud rate (2000000)

### Low frame rate
- Increase `--threshold` to reduce bandwidth
- Lower `--target-fps` if serial can't keep up
- Reduce `--max-updates-per-frame`

### Touch not working
- Ensure TAMC_GT911 library is installed
- Touch is polled during both packet sync and main loop
- Coordinates are inverted to match display orientation

### Colors look wrong
- The display uses RGB565 color format
- BGR/RGB conversion happens in `transmitter.py` via OpenCV

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- [LovyanGFX](https://github.com/lovyan03/LovyanGFX) - RGB parallel display driver
- [TAMC_GT911](https://github.com/tamctec/gt911-arduino) - Touch controller library
- ESP32 Arduino core by Espressif
- [mss](https://github.com/BoboTiG/python-mss) - Cross-platform screen capture
