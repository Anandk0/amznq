/*
 * HUMAN FOLLOWER ROBOT - ESP8266 RECEIVER CODE
 * - Listens for UDP commands from Python script
 * - Controls L298N Motor Driver (4-Pin Logic)
 * - Ultrasonic Safety Stop enabled
 * - UPDATED: Reduced speed to compensate for camera latency
 */

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <NewPing.h>

// ================= CONFIGURATION =================
// !!! CHANGE THESE TO YOUR WIFI CREDENTIALS !!!
const char* ssid = "Poco M6 Pro 5g";
const char* password = "12345679";

// UDP Settings (Must match Python script)
unsigned int udpPort = 8888;

// Motor Speed (0 - 1023)
// REDUCED SPEED to prevent overshooting due to video lag
int SPEED = 200;       // Slower forward speed
int TURN_SPEED = 100;  // Slower turning speed

// Ultrasonic Safety Settings
#define TRIG_PIN D6
#define ECHO_PIN D7
#define MAX_DISTANCE 200 // Maximum distance we want to ping for (in cm)
#define SAFE_DISTANCE 30 // Stop if obstacle is closer than this (in cm)

// ================= PIN DEFINITIONS =================
// L298N Motor Driver Pins
// Motor A (Left)
int ENA = D1; // PWM Speed Control
int IN1 = D3; // Direction 1
int IN2 = D4; // Direction 2

// Motor B (Right)
int ENB = D2; // PWM Speed Control
int IN3 = D5; // Direction 1
int IN4 = D8; // Direction 2

// ================= OBJECTS =================
WiFiUDP udp;
char packetBuffer[255]; // Buffer to hold incoming packet
NewPing sonar(TRIG_PIN, ECHO_PIN, MAX_DISTANCE);

void setup() {
  // 1. Initialize Serial for debugging
  Serial.begin(115200);
  Serial.println("\n--- Robot Booting ---");

  // 2. Initialize Motor Pins
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  
  // Start with motors stopped
  Stop();

  // 3. Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.print("Robot IP Address: ");
  Serial.println(WiFi.localIP()); // <--- USE THIS IP IN YOUR PYTHON SCRIPT

  // 4. Start UDP Listener
  udp.begin(udpPort);
  Serial.printf("Listening for UDP commands on port %d\n", udpPort);
}

void loop() {
  // --- SAFETY CHECK FIRST ---
  // Check distance using Ultrasonic sensor
  int distance = sonar.ping_cm();
  
  // If distance is valid (>0) AND too close (<SAFE_DISTANCE)
  if (distance > 0 && distance < SAFE_DISTANCE) {
    // Only print warning occasionally to avoid spamming logic
    // Force Stop
    Stop();
    return; // Skip the rest of the loop (ignore commands)
  }

  // --- READ UDP COMMANDS ---
  int packetSize = udp.parsePacket();
  if (packetSize) {
    // Read the packet into the buffer
    int len = udp.read(packetBuffer, 255);
    if (len > 0) {
      packetBuffer[len] = 0; // Null-terminate the string
    }
    
    String command = String(packetBuffer);
    Serial.print("CMD Received: ");
    Serial.println(command);

    // --- EXECUTE MOTOR LOGIC ---
    if (command == "FORWARD") {
      Forward();
    }
    else if (command == "LEFT") {
      TurnLeft();
    }
    else if (command == "RIGHT") {
      TurnRight();
    }
    else if (command == "STOP") {
      Stop();
    }
  }
}

// ================= MOTOR FUNCTIONS =================

void Forward() {
  // Motor A Forward
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  // Motor B Forward
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  // Set Speed
  analogWrite(ENA, SPEED);
  analogWrite(ENB, SPEED);
}

void TurnLeft() {
  // Rotate Left in place (Pivot Turn)
  // Motor A Forward
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  // Motor B Backward
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  // Set Speed
  analogWrite(ENA, TURN_SPEED);
  analogWrite(ENB, TURN_SPEED);
}

void TurnRight() {
  // Rotate Right in place (Pivot Turn)
  // Motor A Backward
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  // Motor B Forward
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  // Set Speed
  analogWrite(ENA, TURN_SPEED);
  analogWrite(ENB, TURN_SPEED);
}

void Stop() {
  // Stop logic signals
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  // Cut power
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}