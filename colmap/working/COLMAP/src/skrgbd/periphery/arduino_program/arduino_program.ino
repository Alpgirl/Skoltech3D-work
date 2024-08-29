/* Pins 2-5 are for LED stripes.
 * Pin 7 is for servos.
 * Pins 25:43:2 are for relay switching: 25:37:2 for hard light, and 39:43:2 for softboxes.
 */
#include <FastLED.h>
#include "Arduino.h"
#include "AX12A.h"


#define STRIPS_0 2
#define NUM_STRIPS 4
#define LEDS_N 197

CRGBArray<LEDS_N> leds;
CRGBSet bar0(leds(0, 11));
CRGBSet bar1(leds(12, 49));
CRGBSet bar2(leds(50, 60));
CRGBSet bar3(leds(61, 98));
CRGBSet bar4(leds(99, 109));
CRGBSet bar5(leds(110, 147));
CRGBSet bar6(leds(148, 158));
CRGBSet bar7(leds(159, 196));
CRGBSet bars[8] = {bar0, bar1, bar2, bar3, bar4, bar5, bar6, bar7};
CRGBSet all_leds(leds(0, LEDS_N - 1));

CLEDController *controllers[NUM_STRIPS];


#define SERVO_SERIAL (Serial1)  // Serial port for communication with Dynamixels, default: Serial1
#define DynaBaud (115200)       // Dynamixel Serial baudrate, default: 1000000
#define DirectionPin (7)        // The number of pin for communication direction, default: 7
#define ANGLE_THRESHOLD (10)    // Maximum difference between set and real angle, default: 10
#define MAX_MOVE_ATTEMPTS 2     // Maximum number of attempts to move the servo to a certain position, default: 2
#define MOVE_DELAY (700)        // Time delay for servo movement, default: 700
#define READ_DELAY (500)        // Time delay for reading data from servos, default: 500
#define IR_SERVO_ID 1
#define IR_SERVO_OPEN 512
#define IR_SERVO_CLOSE 819
#define LF_SERVO_ID 2
#define LF_SERVO_OPEN 400
#define LF_SERVO_CLOSE 250
#define LEFT_PHONE_IR_SERVO_ID 4
#define LEFT_PHONE_IR_SERVO_OPEN 580
#define LEFT_PHONE_IR_SERVO_CLOSE 750
#define RIGHT_PHONE_IR_SERVO_ID 3
#define RIGHT_PHONE_IR_SERVO_OPEN 580
#define RIGHT_PHONE_IR_SERVO_CLOSE 750


#define RELAYS_0 25
#define RELAYS_N 11

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(9600);

  all_leds = CRGB::White;
  bar6 = CRGB::Black;
  CRGB correction(255, 62, 250);  // correction that fits our soft/hard LED lamps
  controllers[0] = &FastLED.addLeds<WS2811, STRIPS_0 + 0, GRB>(leds, 50).setCorrection(correction);
  controllers[1] = &FastLED.addLeds<WS2811, STRIPS_0 + 1, GRB>(leds, 50, 49).setCorrection(correction);
  controllers[2] = &FastLED.addLeds<WS2811, STRIPS_0 + 2, GRB>(leds, 99, 49).setCorrection(correction);
  controllers[3] = &FastLED.addLeds<WS2811, STRIPS_0 + 3, GRB>(leds, 148, 49).setCorrection(correction);

  ax12a.begin(DynaBaud, DirectionPin, &SERVO_SERIAL);

  for (int i = 0; i < RELAYS_N; i++) {
    pinMode(RELAYS_0 + i * 2, OUTPUT);
    digitalWrite(RELAYS_0 + i * 2, HIGH);
  }

  for (int i = 0; i < 2; i ++) {
    FastLED.setBrightness(255);
    FastLED.show();
    delay(100);
    FastLED.setBrightness(0);
    FastLED.show();
    delay(100);
  }
  
  Serial.println("ready");
}

void loop() {
  if (Serial.available()) {
    digitalWrite(LED_BUILTIN, HIGH);
    byte msg = Serial.read();
    bool on = bitRead(msg, 0);
    byte id = msg >> 1;
    if (id == 0) {  // kinect IR shutter
      move_servo(IR_SERVO_ID, on ? IR_SERVO_OPEN : IR_SERVO_CLOSE);
    } else if (id == 12) {  // kinect light filter
      move_servo(LF_SERVO_ID, on ? LF_SERVO_CLOSE : LF_SERVO_OPEN);
    } else if (id == 22) {
      move_servo(LEFT_PHONE_IR_SERVO_ID, on ? LEFT_PHONE_IR_SERVO_OPEN : LEFT_PHONE_IR_SERVO_CLOSE);
    } else if (id == 23) {
      move_servo(RIGHT_PHONE_IR_SERVO_ID, on ? RIGHT_PHONE_IR_SERVO_OPEN : RIGHT_PHONE_IR_SERVO_CLOSE);
    } else if (id == 1) {  // all led strips
      FastLED.setBrightness(on ? 255 : 0);
      FastLED.show();
    } else if (id == 21) {  // all led strips with low brightness
      FastLED.setBrightness(on ? 26 : 0);
      FastLED.show();
    } else if (2 <= id && id <= 11) {  // relays
      digitalWrite(RELAYS_0 + (id - 2) * 2, on ? LOW : HIGH);
    }
    digitalWrite(LED_BUILTIN, LOW);
    Serial.println(msg);
  }
}

void move_servo(unsigned char id, int pos){
  int ret;
  for (int attempt_i = 0; attempt_i < MAX_MOVE_ATTEMPTS; attempt_i += 1) {
    ax12a.move(id, pos);
    delay(MOVE_DELAY);
    ret = ax12a.readPositionNew(id);
//    Serial.print(id);
//    Serial.print(": ");
//    Serial.print(ret);
//    Serial.print(" -> ");
//    Serial.println(pos);
    if (abs(ret - pos) < ANGLE_THRESHOLD) return;
    delay(READ_DELAY);
  }
//  Serial.print("Max attempts reached");
}
