// 忍术印术 LED 效果
// 接收网页发送的 "JUTSU:N"，根据忍术编号 N 在 13 号 LED 执行不同效果
//
// 编号 → 忍术           → 效果
//   1  → 火遁·豪火球之术 → 单次长亮 1s（稳重蓄力）
//   2  → 雷遁·千鸟       → 快速四连闪（雷电急促）
//   3  → 土遁·土流壁     → 呼吸渐亮渐灭（厚重缓慢）
//   4  → 水遁·水龙弹     → 双闪（慢，如水波）
//   5  → 风遁·螺旋丸     → SOS 急促闪烁（旋转爆发）

#define LED 13

void setup() {
  pinMode(LED, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (!line.startsWith("JUTSU:")) return;

  int n = line.substring(6).toInt();
  Serial.print("OK:"); Serial.println(n);

  switch (n) {
    case 1: effect1(); break;
    case 2: effect2(); break;
    case 3: effect3(); break;
    case 4: effect4(); break;
    case 5: effect5(); break;
    default: blink(n, 150, 150); break;
  }
}

// 1 火遁·豪火球之术：单次长亮 1 秒
void effect1() {
  digitalWrite(LED, HIGH);
  delay(1000);
  digitalWrite(LED, LOW);
}

// 2 雷遁·千鸟：快速四连闪
void effect2() {
  blink(4, 80, 80);
}

// 3 土遁·土流壁：呼吸渐亮渐灭（13号不支持PWM，用快速开关模拟）
void effect3() {
  for (int i = 0; i < 30; i++) {
    digitalWrite(LED, HIGH); delay(i);
    digitalWrite(LED, LOW);  delay(30 - i);
  }
  for (int i = 30; i > 0; i--) {
    digitalWrite(LED, HIGH); delay(i);
    digitalWrite(LED, LOW);  delay(30 - i);
  }
}

// 4 水遁·水龙弹：双闪（慢）
void effect4() {
  blink(2, 300, 200);
}

// 5 风遁·螺旋丸：SOS（3短 3长 3短）
void effect5() {
  blink(3, 100, 100);
  delay(200);
  blink(3, 300, 100);
  delay(200);
  blink(3, 100, 100);
}

// 工具：闪 n 次，亮 onMs 灭 offMs
void blink(int n, int onMs, int offMs) {
  for (int i = 0; i < n; i++) {
    digitalWrite(LED, HIGH); delay(onMs);
    digitalWrite(LED, LOW);
    if (i < n - 1) delay(offMs);
  }
}
