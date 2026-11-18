class_name VisionBridge
extends Node
## Talks to web/vision/vision.js (window.shinobiVision) in the web build.
## Emits `prediction` once per finished camera check.

signal prediction(gesture: String, conf: float)

var available := OS.has_feature("web")
var phase := "idle"
var progress := 0.0
var error := ""
var ms := 0
var backend := ""

var _js: JavaScriptObject
var _last_seq := 0
var _last_rect := []


func _process(_delta: float) -> void:
	if not available:
		return
	if _js == null:
		# vision.js is a module and may finish loading after the engine starts.
		if not JavaScriptBridge.eval("!!window.shinobiVision", true):
			if phase != "error" and Time.get_ticks_msec() > 30000:
				phase = "error"
				error = "load-failed"
			return
		_js = JavaScriptBridge.get_interface("shinobiVision")
	var st: JavaScriptObject = _js.state
	phase = str(st.phase)
	progress = float(st.progress)
	error = str(st.error)
	ms = int(st.ms)
	backend = str(st.backend)
	var seq := int(st.seq)
	if seq != _last_seq:
		_last_seq = seq
		prediction.emit(str(st.gesture), float(st.conf))


func start() -> void:
	if _js != null:
		_js.start()


## rect is in viewport coordinates; the camera canvas is laid over it.
func set_cam_rect(rect: Rect2, viewport_size: Vector2, visible: bool) -> void:
	if _js == null:
		return
	var key := [rect, viewport_size, visible]
	if key == _last_rect:
		return
	_last_rect = key
	_js.setCamRect(rect.position.x / viewport_size.x, rect.position.y / viewport_size.y,
		rect.size.x / viewport_size.x, rect.size.y / viewport_size.y, visible)
