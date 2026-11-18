class_name Data
## Seals, jutsus, songs and on-screen text. Same content as ninjutsu-v10.html.

## Model class name -> seal kanji (GESTURE_MAP in the original).
const GESTURE_MAP := {
	"Zi": "子", "Chou": "丑", "Yin": "寅", "Mou": "卯",
	"Chen": "辰", "Si": "巳", "Wu": "午", "Wei": "未",
	"Shen": "申", "You": "酉", "Xu": "戌", "Hai": "亥",
}

const SEALS := [
	{"kanji": "子", "romaji": "Ne", "zh": "鼠印", "en": "Rat", "img": "ne"},
	{"kanji": "丑", "romaji": "Ushi", "zh": "牛印", "en": "Ox", "img": "ushi"},
	{"kanji": "寅", "romaji": "Tora", "zh": "虎印", "en": "Tiger", "img": "tora"},
	{"kanji": "卯", "romaji": "U", "zh": "兔印", "en": "Rabbit", "img": "u"},
	{"kanji": "辰", "romaji": "Tatsu", "zh": "龙印", "en": "Dragon", "img": "tatsu"},
	{"kanji": "巳", "romaji": "Mi", "zh": "蛇印", "en": "Snake", "img": "mi"},
	{"kanji": "午", "romaji": "Uma", "zh": "马印", "en": "Horse", "img": "uma"},
	{"kanji": "未", "romaji": "Hitsuji", "zh": "羊印", "en": "Ram", "img": "hitsuji"},
	{"kanji": "申", "romaji": "Saru", "zh": "猴印", "en": "Monkey", "img": "saru"},
	{"kanji": "酉", "romaji": "Tori", "zh": "鸡印", "en": "Bird", "img": "tori"},
	{"kanji": "戌", "romaji": "Inu", "zh": "狗印", "en": "Dog", "img": "inu"},
	{"kanji": "亥", "romaji": "I", "zh": "猪印", "en": "Boar", "img": "i"},
]

const JUTSUS := [
	{"zh": "火遁·豪火球之术", "en": "Fire Style · Great Fireball", "seals": ["子", "午", "寅", "卯", "巳"], "color": Color("#ff4400")},
	{"zh": "雷遁·千鸟", "en": "Lightning Style · Chidori", "seals": ["丑", "寅", "辰", "巳", "子"], "color": Color("#88aaff")},
	{"zh": "土遁·土流壁", "en": "Earth Style · Mud Wall", "seals": ["亥", "寅", "丑", "申", "未"], "color": Color("#aa8844")},
	{"zh": "水遁·水龙弹", "en": "Water Style · Water Dragon", "seals": ["卯", "辰", "午", "未", "酉"], "color": Color("#44aaff")},
	{"zh": "风遁·螺旋丸", "en": "Wind Style · Rasengan", "seals": ["戌", "酉", "申", "午", "寅"], "color": Color("#aaffaa")},
]

## file = name under music/ on the site (see tools/build.py), src = name in the repo's music/.
const SONGS := [
	{"name": "踊", "file": "odoru.mp3", "src": "128bpm踊.mp3", "bpm": 128},
	{"name": "紅蓮華", "file": "gurenge.mp3", "src": "135bpm紅蓮華.mp3", "bpm": 135},
	{"name": "il vento d'oro", "file": "vento.mp3", "src": "136bpmil vento d'oro.mp3", "bpm": 136},
]

const TEXT := {
	"title": ["忍　術　印　術", "N I N J U T S U"],
	"subtitle": ["— 忍术印术识别系统 —", "— NINJUTSU HAND SEAL SYSTEM —"],
	"lang": ["EN", "中文"],
	"practice": ["练　习", "PRACTICE"],
	"challenge": ["挑　战", "CHALLENGE"],
	"music": ["音　楽", "MUSIC"],
	"start": ["启　动", "START"],
	"retry": ["重　试", "RETRY"],
	"reroll": ["换一个忍术", "Another jutsu"],
	"none": ["— 未检测 —", "— no seal —"],
	"timer": ["挑战计时", "TIME LEFT"],
	"st_idle": ["点「连接摄像头」开始", "Press CONNECT CAMERA to begin"],
	"st_camera": ["请在浏览器弹窗中允许使用摄像头…", "Allow camera access in the browser prompt…"],
	"st_loading": ["正在下载识别模型 %d%%（首次约 66 MB）", "Downloading the seal model %d%% (about 66 MB the first time)"],
	"st_ready": ["识别中 — 请做出印术", "Watching — make the seal"],
	"st_ready_ms": ["识别中 — 请做出印术（%d ms）", "Watching — make the seal (%d ms)"],
	"st_debug": ["测试模式：按 1–9、0、-、= 代替手势，空格 = 正确的印", "Test mode: keys 1–9, 0, -, = stand in for the seals; Space = the right seal"],
	"st_web_only": ["摄像头识别只在网页版可用", "Camera recognition runs in the web build only"],
	"err_camera-denied": ["摄像头被拒绝：请在地址栏允许摄像头，再点重试", "Camera blocked: allow it in the address bar, then press RETRY"],
	"err_no-camera": ["没有找到摄像头", "No camera found"],
	"err_load-failed": ["模型加载失败，请刷新页面重试", "The seal model failed to load. Refresh the page and retry"],
	"jutsu_done": ["— 印術成功 —", "— JUTSU COMPLETE —"],
	"bar": ["小节", "BAR"],
	"score": ["得分", "SCORE"],
	"waiting": ["— 等待启动 —", "— press START —"],
	"get_ready": ["— 准备 —", "— get ready —"],
	"loading_song": ["— 正在载入歌曲 —", "— loading song —"],
	"song_failed": ["— 歌曲载入失败 —", "— song failed to load —"],
	"seal_n": ["印　%d / %d", "SEAL %d / %d"],
	"song_over": ["— 演奏结束 —", "— song over —"],
	"perfect": ["— 完美！印术全部成功 %d / %d —", "— PERFECT! %d / %d seals —"],
	"good": ["— 印术成功 %d / %d —", "— JUTSU COMPLETE %d / %d —"],
	"result": ["— 印术结果　%d / %d —", "— RESULT %d / %d —"],
	"game_over": ["音乐模式结束", "MUSIC ROUND OVER"],
	"final": ["得分：%d　　完成：%d / %d", "Score: %d    Rounds: %d / %d"],
	"again": ["再试一次", "AGAIN"],
	"pick_song": ["选歌", "PICK SONG"],
	"go_practice": ["去练习", "PRACTICE"],
	"go_challenge": ["去挑战", "CHALLENGE"],
	"close": ["关　闭", "CLOSE"],
	"web_notice": ["这是网页版。手势识别在你的浏览器里运行，摄像头画面不会上传。\n\n识别最好有显卡（GPU）。有 GPU 的电脑反应很快；没有 GPU 时识别会有延迟，做印时请多保持一会儿，直到打勾。\n\n建议使用电脑上最新版的 Chrome 或 Edge。第一次打开要下载约 66 MB 的识别模型。", "This is the web version. Seal recognition runs in your browser and your camera video is never uploaded.\n\nRecognition works best with a graphics card (GPU). With a GPU it reacts quickly; without one there is a delay, so hold each seal a little longer until it ticks.\n\nUse a current Chrome or Edge on a computer. The first visit downloads the seal model (about 66 MB)."],
	"connect": ["连接摄像头", "CONNECT CAMERA"],
	"connect_debug": ["开始（测试模式）", "START (TEST MODE)"],
	"choose_mode": ["— 选择模式 —", "— CHOOSE A MODE —"],
	"desc_practice": ["没有时间限制，跟着顺序慢慢学每个印", "No timer. Learn each seal at your own pace"],
	"desc_challenge": ["每个印只有 10 秒，超时就从头再来", "10 seconds per seal, or the jutsu starts over"],
	"desc_music": ["跟着歌曲的节拍，每小节做一个印", "One seal per bar, on the beat of the song"],
	"menu": ["菜单", "MENU"],
	"speed_gpu": ["识别速度：每次约 %d ms（正在使用 GPU）", "Recognition: about %d ms per check (using the GPU)"],
	"speed_cpu": ["识别速度：每次约 %d ms（没有可用的 GPU，识别会有延迟）", "Recognition: about %d ms per check (no GPU available, expect a delay)"],
	"how": ["跟着右边的顺序做出每个印，保持一下直到打勾。", "Make each seal in the order on the right and hold it until it ticks."],
}
