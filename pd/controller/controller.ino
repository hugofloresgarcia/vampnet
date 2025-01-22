#include <Adafruit_SSD1327.h>
#include "Adafruit_seesaw.h"
#include "Adafruit_NeoPixel.h"
#include <seesaw_neopixel.h>

#define OLED_RESET -1
#define OLED_ADDR 0x3D
#define ENCODER_BASE_ADDR 0x36
#define BUTTON_BASE_ADDR 0x3A

#define SS_SWITCH 24
#define SS_NEOPIX 6

#define BUTTON1 18
#define BUTTON2 19
#define BUTTON3 20

#define PMW1 12
#define PMW2 13
#define PMW3 0

class SSEncoder {
public:
  Adafruit_seesaw seesaw;
  seesaw_NeoPixel pixel;
  int32_t position;
  bool initialized;

  SSEncoder(uint8_t addr) : 
        seesaw(), 
        pixel(1, SS_NEOPIX, NEO_GRB + NEO_KHZ800), 
        position(0), 
        initialized(false), 
        address(addr) {}

  bool begin() {
    if (!seesaw.begin(address) || !pixel.begin(address)) {
      return false;
    }
    seesaw.pinMode(SS_SWITCH, INPUT_PULLUP);
    pixel.setBrightness(30);
    pixel.show();
    position = seesaw.getEncoderPosition();
    initialized = true;
    return true;
  }

  int32_t getPosition() {
    if (!initialized) return 0;
    return seesaw.getEncoderPosition();
  }

  bool isPressed() {
    if (!initialized) return false;
    return !seesaw.digitalRead(SS_SWITCH);
  }

  void setColor(uint32_t color) {
    if (!initialized) return;
    pixel.setPixelColor(0, color);
    pixel.show();
  }

private:
  uint8_t address;
};

class SSArcadeButton {
public:
  Adafruit_seesaw &seesaw;
  uint8_t pin;
  uint8_t pwmPin;
  bool state;

  SSArcadeButton(Adafruit_seesaw &seesaw, uint8_t pin, uint8_t pwmPin) : 
        seesaw(seesaw), pin(pin), pwmPin(pwmPin), state(true) {}

  void begin() {
    seesaw.pinMode(pin, INPUT_PULLUP);
    seesaw.analogWrite(pwmPin, 127); // Initialize LED with medium brightness
    state = seesaw.digitalRead(pin);
  }

  bool isPressed() {
    return !seesaw.digitalRead(pin);
  }

  bool hasStateChanged() {
    bool newState = seesaw.digitalRead(pin);
    if (newState != state) {
      state = newState;
      return true;
    }
    return false;
  }

  void setBrightness(uint8_t brightness) {
    seesaw.analogWrite(pwmPin, brightness);
  }
};

SSEncoder encoders[4] = {
  SSEncoder(ENCODER_BASE_ADDR),
  SSEncoder(ENCODER_BASE_ADDR + 1),
  SSEncoder(ENCODER_BASE_ADDR + 2),
  SSEncoder(ENCODER_BASE_ADDR + 3)
};

Adafruit_seesaw button_seesaw;
SSArcadeButton buttons[3] = {
  SSArcadeButton(button_seesaw, BUTTON1, PMW1),
  SSArcadeButton(button_seesaw, BUTTON2, PMW2),
  SSArcadeButton(button_seesaw, BUTTON3, PMW3)
};

void setup() {
  Serial.begin(115200);

  Serial.println("Initializing encoders and buttons");
  for (uint8_t i = 0; i < 4; i++) {
    if (!encoders[i].begin()) {
      Serial.print("Failed to initialize encoder ");
      Serial.println(i);
    }
  }

  // Initialize separate seesaw for arcade buttons
  if (!button_seesaw.begin(BUTTON_BASE_ADDR)) {
    Serial.println("Failed to initialize button seesaw");
    while (1) delay(10);
  }

  for (uint8_t i = 0; i < 3; i++) {
    buttons[i].begin();
  }
}

void loop() {
  for (uint8_t i = 0; i < 4; i++) {
    if (!encoders[i].initialized) continue;

    int32_t new_position = encoders[i].getPosition();
    if (encoders[i].position != new_position) {
      encoders[i].position = new_position;
      Serial.print("Encoder ");
      Serial.print(i);
      Serial.print(": ");
      Serial.println(new_position);

      encoders[i].setColor(Wheel((new_position * 4) & 0xFF));
    }

    if (encoders[i].isPressed()) {
      Serial.print("Encoder ");
      Serial.print(i);
      Serial.println(" pressed");
    }
  }

  for (uint8_t i = 0; i < 3; i++) {
    if (buttons[i].hasStateChanged()) {
      Serial.print("Button ");
      Serial.print(i + 1);
      Serial.println(buttons[i].isPressed() ? " pressed" : " released");

      // Adjust brightness based on button state
      buttons[i].setBrightness(buttons[i].isPressed() ? 255 : 0);
    }
  }

  delay(50);
}

uint32_t Wheel(byte WheelPos) {
  WheelPos = 255 - WheelPos;
  if (WheelPos < 85) {
    return Adafruit_NeoPixel::Color(255 - WheelPos * 3, 0, WheelPos * 3);
  }
  if (WheelPos < 170) {
    WheelPos -= 85;
    return Adafruit_NeoPixel::Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
  WheelPos -= 170;
  return Adafruit_NeoPixel::Color(WheelPos * 3, 255 - WheelPos * 3, 0);
}
