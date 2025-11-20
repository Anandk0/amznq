/*
 * ULTIMATE HYBRID ROBOT
 * 1. UDP Listener (For Python AI Tracking)
 * 2. HTTP Web Server (For Wifi RC Car App)
 * 3. Ultrasonic Object Follower (Safety Override)
 */

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <ESP8266WebServer.h>
#include <NewPing.h>

// ===== CONFIGURATION =====
const char* ssid = "Poco M6 Pro 5g";
const char* password = "12345679";
unsigned int udpPort = 8888;

// Tuning
int SPEED_MAX = 700;    // Speed for App/Python
int SPEED_FOLLOW = 550; // Speed for Object Following
int DIST_KEEP = 20;     // cm to keep distance
int RADAR_RANGE = 50;   // cm to trigger object follower mode

// Pins (L298N 4-Pin Logic)
int ENA = D1; 
int ENB = D2; 
int IN1 = D3; int IN2 = D4; // Motor A
int IN3 = D5; int IN4 = D8; // Motor B

// Ultrasonic Pins
const int trigPin = D6;
const int echoPin = D7;

// Globals
WiFiUDP udp;
ESP8266WebServer server(80); // Web server for App
char packetBuffer[255];
NewPing sonar(trigPin, echoPin, 200); // Max distance 200cm

// Last command timestamp (to auto-stop if signal lost)
unsigned long lastCmdTime = 0;

void setup() {
  Serial.begin(115200);
  
  // Motor Pins
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  
  // WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
  Serial.println(WiFi.localIP());

  // Start Services
  udp.begin(udpPort);
  
  // Setup App Handlers
  server.on("/", HTTP_handleRoot);
  server.onNotFound(HTTP_handleRoot);
  server.begin();
}

void loop() {
  // 1. Check Distance (High Priority)
  int distance = sonar.ping_cm();
  
  // === MODE A: OBJECT FOLLOWER (Safety/Auto) ===
  if (distance > 0 && distance < RADAR_RANGE) {
    if (distance < (DIST_KEEP - 5)) { 
      moveBackward(SPEED_FOLLOW);
    }
    else if (distance >= (DIST_KEEP - 5) && distance <= (DIST_KEEP + 5)) {
      stopMotors();
    }
    else {
      moveForward(SPEED_FOLLOW);
    }
    return; // Skip App/UDP check if busy following object
  }

  // === MODE B: APP CONTROL (HTTP) ===
  server.handleClient();

  // === MODE C: PYTHON AI CONTROL (UDP) ===
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, 255);
    if (len > 0) packetBuffer[len] = 0;
    String cmd = String(packetBuffer);
    
    if (cmd == "FORWARD") moveForward(SPEED_MAX);
    else if (cmd == "LEFT") turnLeft(SPEED_MAX);
    else if (cmd == "RIGHT") turnRight(SPEED_MAX);
    else if (cmd == "STOP") stopMotors();
    
    lastCmdTime = millis();
  }
}

// --- APP HANDLER ---
void HTTP_handleRoot() {
  server.send(200, "text/html", "Robot Online");
  
  if (server.hasArg("State")) {
    String cmd = server.arg("State");
    // Map App Commands to Functions
    if (cmd == "F") moveForward(SPEED_MAX);
    else if (cmd == "B") moveBackward(SPEED_MAX);
    else if (cmd == "L") turnLeft(SPEED_MAX);
    else if (cmd == "R") turnRight(SPEED_MAX);
    else if (cmd == "S") stopMotors();
    
    // Note: The App usually sends commands continuously or sends 'S' on release.
    lastCmdTime = millis();
  }
}

// --- MOVEMENT FUNCTIONS ---
void moveForward(int spd) {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, spd); analogWrite(ENB, spd);
}
void moveBackward(int spd) {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  analogWrite(ENA, spd); analogWrite(ENB, spd);
}
void turnLeft(int spd) {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  analogWrite(ENA, spd); analogWrite(ENB, spd);
}
void turnRight(int spd) {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  analogWrite(ENA, spd); analogWrite(ENB, spd);
}
void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  analogWrite(ENA, 0); analogWrite(ENB, 0);
}