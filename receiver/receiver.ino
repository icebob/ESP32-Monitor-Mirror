/*
 * Pixel Update Receiver for Elecrow CrowPanel ESP32-S3 7.0" (800x480)
 * Uses LovyanGFX with RGB parallel bus (EK9716 driver)
 * Receives data over USB Serial (no WiFi needed)
 *
 * Run-length protocol v2 (little-endian):
 *   Header: 'P' 'X' 'U' 'R' (4 bytes) + version (1 byte, 0x02) + frame_id (uint32 LE) + count (uint16)
 *   Body:   count entries of: y (uint16 LE), x0 (uint16 LE), length (uint16 LE), color (uint16 LE) = 8 bytes each
 *
 * Board settings in Arduino IDE:
 *   Board: ESP32S3 Dev Module
 *   Flash Size: 4MB
 *   Partition Scheme: Huge APP (3MB No OTA/1MB SPIFFS)
 *   PSRAM: OPI PSRAM
 *   USB CDC On Boot: Enabled
 */

#include <LovyanGFX.hpp>
#include <lgfx/v1/platforms/esp32s3/Panel_RGB.hpp>
#include <lgfx/v1/platforms/esp32s3/Bus_RGB.hpp>
#include <esp_heap_caps.h>
#include <Wire.h>
#include <TAMC_GT911.h>

// --- LovyanGFX display configuration for CrowPanel 7.0" ---
class LGFX : public lgfx::LGFX_Device {
public:
  lgfx::Bus_RGB   _bus_instance;
  lgfx::Panel_RGB _panel_instance;

  LGFX(void) {
    {
      auto cfg = _bus_instance.config();
      cfg.panel = &_panel_instance;

      cfg.pin_d0  = GPIO_NUM_15; // B0
      cfg.pin_d1  = GPIO_NUM_7;  // B1
      cfg.pin_d2  = GPIO_NUM_6;  // B2
      cfg.pin_d3  = GPIO_NUM_5;  // B3
      cfg.pin_d4  = GPIO_NUM_4;  // B4

      cfg.pin_d5  = GPIO_NUM_9;  // G0
      cfg.pin_d6  = GPIO_NUM_46; // G1
      cfg.pin_d7  = GPIO_NUM_3;  // G2
      cfg.pin_d8  = GPIO_NUM_8;  // G3
      cfg.pin_d9  = GPIO_NUM_16; // G4
      cfg.pin_d10 = GPIO_NUM_1;  // G5

      cfg.pin_d11 = GPIO_NUM_14; // R0
      cfg.pin_d12 = GPIO_NUM_21; // R1
      cfg.pin_d13 = GPIO_NUM_47; // R2
      cfg.pin_d14 = GPIO_NUM_48; // R3
      cfg.pin_d15 = GPIO_NUM_45; // R4

      cfg.pin_henable = GPIO_NUM_41;
      cfg.pin_vsync   = GPIO_NUM_40;
      cfg.pin_hsync   = GPIO_NUM_39;
      cfg.pin_pclk    = GPIO_NUM_0;
      cfg.freq_write  = 15000000;

      cfg.hsync_polarity    = 0;
      cfg.hsync_front_porch = 40;
      cfg.hsync_pulse_width = 48;
      cfg.hsync_back_porch  = 40;

      cfg.vsync_polarity    = 0;
      cfg.vsync_front_porch = 1;
      cfg.vsync_pulse_width = 31;
      cfg.vsync_back_porch  = 13;

      cfg.pclk_active_neg = 1;
      cfg.de_idle_high    = 0;
      cfg.pclk_idle_high  = 0;

      _bus_instance.config(cfg);
    }
    {
      auto cfg = _panel_instance.config();
      cfg.memory_width  = 800;
      cfg.memory_height = 480;
      cfg.panel_width   = 800;
      cfg.panel_height  = 480;
      cfg.offset_x = 0;
      cfg.offset_y = 0;
      _panel_instance.config(cfg);
    }
    _panel_instance.setBus(&_bus_instance);
    setPanel(&_panel_instance);
  }
};

LGFX tft;

// Display dimensions
#define DISPLAY_WIDTH  800
#define DISPLAY_HEIGHT 480

// Touch (GT911)
#define TOUCH_SDA 19
#define TOUCH_SCL 20
TAMC_GT911 ts(TOUCH_SDA, TOUCH_SCL, -1, -1, DISPLAY_WIDTH, DISPLAY_HEIGHT);
bool wasTouched = false;

// Backlight
#define TFT_BL 2

// Protocol constants
const uint8_t MAGIC_RUN[4] = {'P', 'X', 'U', 'R'};
const uint8_t RUN_VERSION   = 0x02;

// Stats
unsigned long frameCount = 0;
unsigned long updatesApplied = 0;

// Raw byte buffer for bulk reads
uint8_t* rawBuffer = nullptr;
uint32_t rawBufferSize = 0;

bool ensureRawBuffer(uint32_t needed) {
  if (needed <= rawBufferSize && rawBuffer != nullptr) {
    return true;
  }
  uint8_t* tmp = (uint8_t*)ps_malloc(needed);
  if (!tmp) tmp = (uint8_t*)malloc(needed);
  if (!tmp) return false;
  if (rawBuffer) free(rawBuffer);
  rawBuffer = tmp;
  rawBufferSize = needed;
  return true;
}

bool readExactly(uint8_t* dst, size_t len) {
  size_t got = 0;
  unsigned long start = millis();
  while (got < len) {
    int avail = Serial.available();
    if (avail > 0) {
      int toRead = min((size_t)avail, len - got);
      int chunk = Serial.readBytes(dst + got, toRead);
      got += chunk;
      start = millis();  // reset timeout on progress
    } else {
      if (millis() - start > 5000) {
        return false;  // 5s timeout with no data
      }
      delayMicroseconds(100);
    }
  }
  return true;
}

void showWaitingScreen() {
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setCursor(20, 40);
  tft.setTextSize(3);
  tft.println("Pixel RX");
  tft.setCursor(20, 100);
  tft.setTextSize(2);
  tft.println("Serial Mode");
  tft.setCursor(20, 140);
  tft.setTextSize(2);
  tft.setTextColor(TFT_GREEN, TFT_BLACK);
  tft.println("Waiting for data...");
}

void setup() {
  Serial.begin(2000000);
  Serial.setRxBufferSize(65536);  // large receive buffer
  delay(500);

  // Backlight on via LEDC PWM
  ledcAttach(TFT_BL, 300, 8);
  ledcWrite(TFT_BL, 255);

  tft.init();

  // Init touch
  Wire.begin(TOUCH_SDA, TOUCH_SCL);
  ts.begin();
  ts.setRotation(ROTATION_NORMAL);

  showWaitingScreen();
}

void handleTouch();  // forward declaration

// Scan for magic bytes in the serial stream, check touch while waiting
bool syncToMagic() {
  uint8_t buf[4] = {0};
  int pos = 0;

  unsigned long start = millis();
  unsigned long lastTouch = 0;
  while (millis() - start < 5000) {
    // Check touch every ~20ms while waiting for data
    if (millis() - lastTouch > 20) {
      handleTouch();
      lastTouch = millis();
    }
    if (Serial.available()) {
      buf[pos % 4] = Serial.read();
      pos++;
      if (pos >= 4) {
        int s = pos % 4;
        if (buf[s] == MAGIC_RUN[0] &&
            buf[(s+1)%4] == MAGIC_RUN[1] &&
            buf[(s+2)%4] == MAGIC_RUN[2] &&
            buf[(s+3)%4] == MAGIC_RUN[3]) {
          return true;
        }
      }
    } else {
      delayMicroseconds(100);
    }
  }
  return false;
}

void handlePacket() {
  // Read rest of header: version(1) + frame_id(4) + count(2) = 7 bytes
  uint8_t hdr[7];
  if (!readExactly(hdr, 7)) return;

  if (hdr[0] != RUN_VERSION) return;

  uint32_t frameId = ((uint32_t)hdr[1]) | ((uint32_t)hdr[2] << 8) |
                     ((uint32_t)hdr[3] << 16) | ((uint32_t)hdr[4] << 24);
  uint16_t count = hdr[5] | (hdr[6] << 8);

  if (count == 0) {
    Serial.write(0x06);  // ACK
    frameCount++;
    return;
  }
  if (count > 60000) return;  // sanity check

  uint32_t payloadSize = (uint32_t)count * 8;
  if (!ensureRawBuffer(payloadSize)) return;

  if (!readExactly(rawBuffer, payloadSize)) return;

  // Draw
  tft.startWrite();
  uint8_t* p = rawBuffer;
  for (uint16_t i = 0; i < count; i++, p += 8) {
    uint16_t y      = p[0] | (p[1] << 8);
    uint16_t x0     = p[2] | (p[3] << 8);
    uint16_t runLen = p[4] | (p[5] << 8);
    uint16_t color  = p[6] | (p[7] << 8);
    if (x0 < DISPLAY_WIDTH && y < DISPLAY_HEIGHT && runLen > 0 && (x0 + runLen) <= DISPLAY_WIDTH) {
      tft.writeFastHLine(x0, y, runLen, color);
      updatesApplied += runLen;
    }
  }
  tft.endWrite();

  // Send ACK
  Serial.write(0x06);

  frameCount++;
}

void sendTouchEvent(uint16_t x, uint16_t y, uint8_t type) {
  // Protocol: 'T' 'C' 'H' + x(uint16 LE) + y(uint16 LE) + type(uint8) = 8 bytes
  // type: 0=press, 1=move, 2=release
  uint8_t pkt[8] = {'T', 'C', 'H',
    (uint8_t)(x & 0xFF), (uint8_t)(x >> 8),
    (uint8_t)(y & 0xFF), (uint8_t)(y >> 8),
    type};
  Serial.write(pkt, 8);
}

void handleTouch() {
  ts.read();
  if (ts.isTouched) {
    uint16_t x = (DISPLAY_WIDTH - 1) - ts.points[0].x;
    uint16_t y = (DISPLAY_HEIGHT - 1) - ts.points[0].y;
    if (!wasTouched) {
      sendTouchEvent(x, y, 0);  // press
    } else {
      sendTouchEvent(x, y, 1);  // move
    }
    wasTouched = true;
  } else if (wasTouched) {
    sendTouchEvent(0, 0, 2);  // release
    wasTouched = false;
  }
}

void loop() {
  if (syncToMagic()) {
    handlePacket();
  }
  handleTouch();
}
