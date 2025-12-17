/*
 * Pixel Update Receiver for ESP32 T-Display (ST7789 135x240)
 * Receives per-pixel updates (x, y, RGB565) over TCP and applies them.
 * Protocol v2 (little-endian):
 *   Header: 'P' 'X' 'U' 'P' (4 bytes) + version (1 byte, 0x02) + frame_id (uint32 LE) + count (uint16)
 *   Body:   count entries of: x (uint8), y (uint8), color (uint16 LE)
 *
 * Optimized for high frame rates with:
 * - Fast SPI clock (80MHz default, configurable)
 * - DMA support for efficient display updates
 * - Run-length encoding support for reduced bandwidth
 */

#include <TFT_eSPI.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <esp_heap_caps.h>  // for PSRAM allocations

#define TFT_MADCTL 0x36
#define TFT_MADCTL_RGB 0x00
#define TFT_MADCTL_BGR 0x08

TFT_eSPI tft = TFT_eSPI();

// Display dimensions
#define DISPLAY_WIDTH 135
#define DISPLAY_HEIGHT 240

// Try the fastest stable SPI clock for the panel; lower to 40000000 if unstable
const uint32_t SPI_TARGET_FREQ = 80000000;

// WiFi credentials - UPDATE THESE WITH YOUR NETWORK
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Network settings
WiFiServer server(8090);  // dedicated port for pixel updates
WiFiClient client;

// Protocol constants (v2 adds frame_id to the header)
const uint8_t MAGIC[4] = {'P', 'X', 'U', 'P'};
const uint8_t PROTO_VERSION = 0x02;
const size_t HEADER_SIZE = 11;  // MAGIC (4) + version (1) + frame_id (4) + count (2)
const uint8_t MAGIC_RUN[4] = {'P', 'X', 'U', 'R'};
const uint8_t RUN_VERSION = 0x01;
const size_t RUN_HEADER_SIZE = 11;  // MAGIC_RUN (4) + version (1) + frame_id (4) + count (2)

// Color configuration (adjust if colors appear swapped)
bool swapBytesSetting = false;  // keep false; colors are provided as RGB565 little-endian
bool useBgrSetting   = true;    // many ST7789 panels are BGR wired

// Stats
unsigned long frameCount = 0;
unsigned long lastStats = 0;
unsigned long updatesApplied = 0;
uint32_t lastFrameId = 0;

struct PixelUpdate {
  uint8_t x;
  uint8_t y;
  uint8_t len;    // for run packets
  uint16_t color;
};

PixelUpdate* updateBuffer = nullptr;
uint32_t bufferCapacity = 0;
bool dmaEnabled = false;

bool ensureUpdateBuffer(uint32_t needed) {
  if (needed <= bufferCapacity && updateBuffer != nullptr) {
    return true;
  }
  PixelUpdate* tmp = (PixelUpdate*)ps_malloc(needed * sizeof(PixelUpdate));
  if (!tmp) {
    tmp = (PixelUpdate*)malloc(needed * sizeof(PixelUpdate));
  }
  if (!tmp) {
    Serial.println("Failed to allocate update buffer");
    return false;
  }
  if (updateBuffer) {
    free(updateBuffer);
  }
  updateBuffer = tmp;
  bufferCapacity = needed;
  return true;
}

bool readExactly(WiFiClient& c, uint8_t* dst, size_t len) {
  size_t got = 0;
  while (got < len && c.connected()) {
    int chunk = c.read(dst + got, len - got);
    if (chunk > 0) {
      got += chunk;
    } else {
      delay(1);  // allow other tasks
    }
  }
  return got == len;
}

void applyColorConfig() {
  tft.setSwapBytes(swapBytesSetting);
  tft.writecommand(TFT_MADCTL);
  tft.writedata(useBgrSetting ? TFT_MADCTL_BGR : TFT_MADCTL_RGB);
}

void showWaitingScreen() {
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setCursor(10, 20);
  tft.setTextSize(2);
  tft.println("Pixel RX");
  tft.setCursor(10, 50);
  tft.setTextSize(1);
  tft.println("IP Address:");
  tft.setCursor(10, 70);
  tft.setTextSize(2);
  tft.setTextColor(TFT_GREEN, TFT_BLACK);
  tft.println(WiFi.localIP().toString());
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setCursor(10, 100);
  tft.setTextSize(1);
  tft.println("Waiting for");
  tft.setCursor(10, 115);
  tft.println("connection...");
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== Pixel Update Receiver ===");

  pinMode(4, OUTPUT);
  digitalWrite(4, HIGH);  // backlight
  tft.init();
  SPI.setFrequency(SPI_TARGET_FREQ);
  dmaEnabled = tft.initDMA();
  tft.setRotation(0);  // portrait
  applyColorConfig();
  tft.fillScreen(TFT_BLACK);

  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(250);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi connection failed");
    tft.fillScreen(TFT_RED);
    tft.setTextColor(TFT_WHITE, TFT_RED);
    tft.setCursor(10, 50);
    tft.setTextSize(2);
    tft.println("WiFi FAILED!");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("\nWiFi connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  showWaitingScreen();

  server.begin();
  server.setNoDelay(true);
  Serial.println("Server listening on port 8090");
}

bool handleClient() {
  // Accept new client
  if (!client || !client.connected()) {
    client = server.available();
    if (client) {
      Serial.println("Client connected");
      client.setNoDelay(true);
      client.setTimeout(50);  // short timeout for reads
      frameCount = 0;
      updatesApplied = 0;
      tft.fillScreen(TFT_BLACK);
    }
  }

  if (!client || !client.connected()) {
    return false;
  }

  // Require header to begin processing (pixel or run)
  if (client.available() < 11) {
    return true;  // keep connection, wait for more data
  }

  // Peek magic to decide packet type
  uint8_t magicBuf[4];
  if (!readExactly(client, magicBuf, 4)) {
    client.stop();
    return false;
  }
  bool isRun = (memcmp(magicBuf, MAGIC_RUN, 4) == 0);
  bool isPixel = (memcmp(magicBuf, MAGIC, 4) == 0);

  if (!isRun && !isPixel) {
    Serial.println("Bad magic; flushing stream");
    client.stop();
    return false;
  }

  if (isPixel) {
    uint8_t rest[HEADER_SIZE - 4];
    if (!readExactly(client, rest, sizeof(rest))) {
      Serial.println("Failed to read pixel header; dropping client");
      client.stop();
      return false;
    }
    if (rest[0] != PROTO_VERSION) {
      Serial.print("Unsupported pixel version: ");
      Serial.println(rest[0], HEX);
      client.stop();
      return false;
    }

    uint32_t frameId = ((uint32_t)rest[1]) | ((uint32_t)rest[2] << 8) | ((uint32_t)rest[3] << 16) | ((uint32_t)rest[4] << 24);
    uint16_t count = rest[5] | (rest[6] << 8);  // little-endian
    if (count == 0) {
      frameCount++;
      lastFrameId = frameId;
      return true;
    }
    if (count > (DISPLAY_WIDTH * DISPLAY_HEIGHT)) {
      Serial.print("Update count too large: ");
      Serial.println(count);
      client.stop();
      return false;
    }

    if (!ensureUpdateBuffer(count)) {
      Serial.println("No buffer for updates; dropping client");
      client.stop();
      return false;
    }

    uint8_t entry[4];
    for (uint16_t i = 0; i < count; i++) {
      if (!readExactly(client, entry, 4)) {
        Serial.println("Stream ended mid-frame; dropping client");
        client.stop();
        return false;
      }
      updateBuffer[i].x = entry[0];
      updateBuffer[i].y = entry[1];
      updateBuffer[i].color = entry[2] | (entry[3] << 8);
    }

    // Apply all updates in one batch after the full frame is received
    tft.startWrite();
    for (uint16_t i = 0; i < count; i++) {
      uint8_t x = updateBuffer[i].x;
      uint8_t y = updateBuffer[i].y;
      if (x < DISPLAY_WIDTH && y < DISPLAY_HEIGHT) {
        tft.setAddrWindow(x, y, 1, 1);
        tft.writeColor(updateBuffer[i].color, 1);
        updatesApplied++;
      }
    }
    tft.endWrite();

    frameCount++;
    lastFrameId = frameId;
    unsigned long now = millis();
    if (now - lastStats > 2000) {
      Serial.print("Frames: ");
      Serial.print(frameCount);
      Serial.print(" (last frameId ");
      Serial.print(lastFrameId);
      Serial.print(") | Updates applied: ");
      Serial.println(updatesApplied);
      lastStats = now;
    }
    return true;
  }

  // Run packet
  uint8_t rest[RUN_HEADER_SIZE - 4];
  if (!readExactly(client, rest, sizeof(rest))) {
    Serial.println("Failed to read run header; dropping client");
    client.stop();
    return false;
  }
  if (rest[0] != RUN_VERSION) {
    Serial.print("Unsupported run version: ");
    Serial.println(rest[0], HEX);
    client.stop();
    return false;
  }

  uint32_t frameId = ((uint32_t)rest[1]) | ((uint32_t)rest[2] << 8) | ((uint32_t)rest[3] << 16) | ((uint32_t)rest[4] << 24);
  uint16_t count = rest[5] | (rest[6] << 8);  // number of runs
  if (count == 0) {
    frameCount++;
    lastFrameId = frameId;
    return true;
  }
  if (count > (DISPLAY_WIDTH * DISPLAY_HEIGHT)) {
    Serial.print("Run count too large: ");
    Serial.println(count);
    client.stop();
    return false;
  }

  if (!ensureUpdateBuffer(count)) {
    Serial.println("No buffer for run updates; dropping client");
    client.stop();
    return false;
  }

  // Each run entry: y (1), x0 (1), length (1), color (2) = 5 bytes
  uint8_t entry[5];
  for (uint16_t i = 0; i < count; i++) {
    if (!readExactly(client, entry, 5)) {
      Serial.println("Stream ended mid-run frame; dropping client");
      client.stop();
      return false;
    }
    updateBuffer[i].y = entry[0];
    updateBuffer[i].x = entry[1];
    updateBuffer[i].len = entry[2];
    updateBuffer[i].color = entry[3] | (entry[4] << 8);
  }

  // Apply runs in one batch
  tft.startWrite();
  for (uint16_t i = 0; i < count; i++) {
    uint8_t x0 = updateBuffer[i].x;
    uint8_t y = updateBuffer[i].y;
    uint8_t runLen = updateBuffer[i].len;
    if (x0 < DISPLAY_WIDTH && y < DISPLAY_HEIGHT && runLen > 0 && (x0 + runLen) <= DISPLAY_WIDTH) {
      tft.setAddrWindow(x0, y, runLen, 1);
      if (dmaEnabled) {
        tft.pushBlock(updateBuffer[i].color, runLen);
      } else {
        tft.writeColor(updateBuffer[i].color, runLen);
      }
      updatesApplied += runLen;
    }
  }
  tft.endWrite();

  frameCount++;
  lastFrameId = frameId;
  unsigned long now = millis();
  if (now - lastStats > 2000) {
    Serial.print("Frames: ");
    Serial.print(frameCount);
    Serial.print(" (last frameId ");
    Serial.print(lastFrameId);
    Serial.print(") | Updates applied: ");
    Serial.println(updatesApplied);
    lastStats = now;
  }

  return true;
}

void loop() {
  handleClient();
  if (client && !client.connected()) {
    Serial.println("Client disconnected");
    showWaitingScreen();
  }
  delay(1);
}

