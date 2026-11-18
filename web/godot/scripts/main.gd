extends Control
## Project Shinobi: make the hand seals of a jutsu in order.
## Modes and rules follow ninjutsu-v10.html: practice, challenge (10 s per
## seal) and music (one seal per bar, 5 jutsus a game).

const THRESHOLD := 0.4
const HOLD_FRAMES := 2
const CHALLENGE_SECONDS := 10.0
const MUSIC_MAX_ROUNDS := 5
const MUSIC_VOLUME := 0.1

const ORANGE := Color("#ff6b00")
const GOLD := Color("#e8d5a3")
const DIM := Color("#7a6a4a")
const PANEL_BG := Color(0.05, 0.05, 0.08, 0.85)
const LINE := Color("#3a2a1a")
const GREEN := Color("#44aa44")
const RED := Color("#cc3333")
const YELLOW := Color("#ffcc44")

var lang := 0  # 0 = Chinese, 1 = English
var debug := false
var running := false  # camera + model ready (or debug keys)
var mode := "practice"
var gen := 0  # bumped on mode change so stale timers do nothing

var jutsu: Dictionary = {}
var jutsu_idx := 0
var step := 0
var hold := 0
var last_seen := ""
var celebrating := false
var resetting := false

var ch_active := false
var ch_left := 0.0

var song_idx := randi() % Data.SONGS.size()
var m_playing := false
var m_starting := false
var m_bar_sec := 2.0
var m_seal_active := false
var m_seal_t := 0.0
var m_seal_hit := false
var m_results: Array[bool] = []
var m_round_pending := false
var m_rounds := 0
var m_score := 0
var m_streak := 0
var m_bar_num := 0

var base_url := ""
var _audio_cache := {}
var _seal_tex := {}
var _font: Font
var _bold: Font

var vision: VisionBridge
var music_player: AudioStreamPlayer

# UI
var title_lbl: Label
var subtitle_lbl: Label
var lang_btn: Button
var mode_box: HBoxContainer
var mode_btns := {}
var song_box: HBoxContainer
var song_btns: Array[Button] = []
var timer_box: VBoxContainer
var timer_lbl: Label
var timer_bar: ProgressBar
var left_box: VBoxContainer
var cam_frame: Panel
var cam_hint: Label
var gesture_lbl: Label
var conf_lbl: Label
var start_btn: Button
var status_lbl: Label
var stack_box: VBoxContainer
var cards: Array[Dictionary] = []  # [next2, next1, current]
var right_box: VBoxContainer
var jutsu_lbl: Label
var progress_bar: ProgressBar
var rows: Array[Dictionary] = []
var reroll_btn: Button
var how_lbl: Label
var hud: PanelContainer
var hud_bar_lbl: Label
var hud_score_lbl: Label
var beat_dots: Array[Panel] = []
var hud_progress: ProgressBar
var hud_phase: Label
var flash: ColorRect
var overlay: Control
var overlay_title: Label
var overlay_sub: Label
var overlay_btns: HBoxContainer
var countdown_lbl: Label
var modal: Control
var modal_kanji: Label
var modal_name: Label
var modal_tex: TextureRect
var modal_close: Button
var fx_layer: Control
var jutsu_particles: CPUParticles2D


func _ready() -> void:
	randomize()
	song_idx = randi() % Data.SONGS.size()
	lang = 0 if OS.get_locale_language() == "zh" else 1
	if OS.has_feature("web"):
		base_url = str(JavaScriptBridge.eval("new URL('.', location.href).href", true))
		debug = str(JavaScriptBridge.eval("location.search", true)).contains("debug")
	else:
		debug = true
	_font = load("res://assets/fonts/serif.ttf")
	_bold = load("res://assets/fonts/serif-bold.ttf")
	var th := Theme.new()
	th.default_font = _font
	th.default_font_size = 18
	theme = th
	for s in Data.SEALS:
		_seal_tex[s.kanji] = load("res://assets/seals/%s.png" % s.img)

	vision = VisionBridge.new()
	add_child(vision)
	vision.prediction.connect(_on_prediction)
	music_player = AudioStreamPlayer.new()
	music_player.volume_db = linear_to_db(MUSIC_VOLUME)
	music_player.finished.connect(_on_song_finished)
	add_child(music_player)

	_build_ui()
	get_viewport().size_changed.connect(_layout)
	_apply_lang()
	set_mode("practice")
	_layout()
	for i in range(1, 8):
		_load_audio("audio/%d.mp3" % i, "audio/%d.mp3" % i)


# ───────────────────────── UI construction ─────────────────────────

func _style(bg: Color, border: Color, width := 1, radius := 4) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = bg
	sb.border_color = border
	sb.set_border_width_all(width)
	sb.set_corner_radius_all(radius)
	sb.content_margin_left = 12
	sb.content_margin_right = 12
	sb.content_margin_top = 6
	sb.content_margin_bottom = 6
	return sb


func _label(size: int, color: Color, bold := false, align := HORIZONTAL_ALIGNMENT_CENTER) -> Label:
	var l := Label.new()
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", color)
	if bold:
		l.add_theme_font_override("font", _bold)
	l.horizontal_alignment = align
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return l


func _glow(l: Label, color: Color, size := 20) -> void:
	l.add_theme_color_override("font_shadow_color", Color(color, 0.55))
	l.add_theme_constant_override("shadow_outline_size", size)
	l.add_theme_constant_override("shadow_offset_x", 0)
	l.add_theme_constant_override("shadow_offset_y", 0)


func _button(cb: Callable, size := 18) -> Button:
	var b := Button.new()
	b.add_theme_font_size_override("font_size", size)
	b.add_theme_font_override("font", _bold)
	b.add_theme_color_override("font_color", GOLD)
	b.add_theme_color_override("font_hover_color", Color.WHITE)
	b.add_theme_color_override("font_pressed_color", Color.WHITE)
	b.add_theme_color_override("font_disabled_color", DIM)
	b.add_theme_stylebox_override("normal", _style(Color(0.05, 0.05, 0.08, 0.8), Color("#5a3a1a")))
	b.add_theme_stylebox_override("hover", _style(Color(0.15, 0.07, 0.02, 0.9), ORANGE))
	b.add_theme_stylebox_override("pressed", _style(Color(0.3, 0.12, 0.0, 0.95), ORANGE))
	b.add_theme_stylebox_override("disabled", _style(Color(0.05, 0.05, 0.08, 0.6), LINE))
	b.add_theme_stylebox_override("focus", StyleBoxEmpty.new())
	b.focus_mode = Control.FOCUS_NONE
	b.mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND
	b.pressed.connect(cb)
	return b


func _set_active(b: Button, on: bool) -> void:
	if on:
		b.add_theme_stylebox_override("normal", _style(Color(0.8, 0.27, 0.0, 0.95), Color("#ff9944")))
		b.add_theme_color_override("font_color", Color.WHITE)
	else:
		b.add_theme_stylebox_override("normal", _style(Color(0.05, 0.05, 0.08, 0.8), Color("#5a3a1a")))
		b.add_theme_color_override("font_color", GOLD)


func _make_card(size: int, alpha: float) -> Dictionary:
	var p := Panel.new()
	p.custom_minimum_size = Vector2(size, size)
	p.add_theme_stylebox_override("panel", _style(Color(0.03, 0.03, 0.06, 0.9), Color(ORANGE, 0.45), 2, 6))
	p.modulate.a = alpha
	p.pivot_offset = Vector2(size, size) / 2
	p.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var t := TextureRect.new()
	t.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	t.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	t.set_anchors_preset(Control.PRESET_FULL_RECT)
	t.offset_left = 4
	t.offset_top = 4
	t.offset_right = -4
	t.offset_bottom = -4
	t.mouse_filter = Control.MOUSE_FILTER_IGNORE
	p.add_child(t)
	return {"panel": p, "tex": t}


func _build_ui() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)

	var bg := TextureRect.new()
	bg.texture = load("res://assets/background.jpg")
	bg.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	bg.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_COVERED
	bg.set_anchors_preset(Control.PRESET_FULL_RECT)
	bg.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(bg)
	var shade := ColorRect.new()
	shade.color = Color(0.02, 0.02, 0.04, 0.45)
	shade.set_anchors_preset(Control.PRESET_FULL_RECT)
	shade.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(shade)

	title_lbl = _label(40, ORANGE, true)
	_glow(title_lbl, ORANGE, 10)
	add_child(title_lbl)
	subtitle_lbl = _label(14, DIM)
	add_child(subtitle_lbl)

	lang_btn = _button(_toggle_lang, 16)
	add_child(lang_btn)

	mode_box = HBoxContainer.new()
	mode_box.add_theme_constant_override("separation", 10)
	for m in ["practice", "challenge", "music"]:
		var b := _button(set_mode.bind(m), 20)
		b.custom_minimum_size = Vector2(150, 44)
		mode_btns[m] = b
		mode_box.add_child(b)
	add_child(mode_box)

	song_box = HBoxContainer.new()
	song_box.add_theme_constant_override("separation", 8)
	for i in Data.SONGS.size():
		var b := _button(select_song.bind(i), 16)
		b.text = Data.SONGS[i].name
		b.custom_minimum_size = Vector2(140, 36)
		song_btns.append(b)
		song_box.add_child(b)
	add_child(song_box)

	timer_box = VBoxContainer.new()
	timer_lbl = _label(14, DIM)
	timer_box.add_child(timer_lbl)
	timer_bar = _bar(Vector2(420, 10), ORANGE)
	timer_box.add_child(timer_bar)
	add_child(timer_box)

	# Left: camera, detected seal, start.
	left_box = VBoxContainer.new()
	left_box.add_theme_constant_override("separation", 10)
	cam_frame = Panel.new()
	cam_frame.custom_minimum_size = Vector2(400, 300)
	cam_frame.add_theme_stylebox_override("panel", _style(Color(0.05, 0.05, 0.08, 0.95), Color(ORANGE, 0.4), 2))
	cam_hint = _label(16, DIM)
	cam_hint.set_anchors_preset(Control.PRESET_FULL_RECT)
	cam_hint.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	cam_hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	cam_frame.add_child(cam_hint)
	left_box.add_child(cam_frame)
	var detect := PanelContainer.new()
	detect.add_theme_stylebox_override("panel", _style(PANEL_BG, LINE))
	var dh := HBoxContainer.new()
	gesture_lbl = _label(22, Color("#ff9944"), true, HORIZONTAL_ALIGNMENT_LEFT)
	gesture_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	conf_lbl = _label(16, DIM, false, HORIZONTAL_ALIGNMENT_RIGHT)
	dh.add_child(gesture_lbl)
	dh.add_child(conf_lbl)
	detect.add_child(dh)
	left_box.add_child(detect)
	start_btn = _button(_on_start, 22)
	start_btn.custom_minimum_size = Vector2(400, 50)
	start_btn.add_theme_stylebox_override("normal", _style(Color("#cc4400"), Color("#ff6b00"), 1))
	start_btn.add_theme_color_override("font_color", Color.WHITE)
	left_box.add_child(start_btn)
	status_lbl = _label(15, Color("#a89470"))
	status_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status_lbl.custom_minimum_size = Vector2(400, 0)
	left_box.add_child(status_lbl)
	add_child(left_box)

	# Centre: next two seals above the current one.
	stack_box = VBoxContainer.new()
	stack_box.alignment = BoxContainer.ALIGNMENT_END
	stack_box.add_theme_constant_override("separation", 12)
	for spec in [[100, 0.5], [140, 0.85], [290, 1.0]]:
		var c := _make_card(spec[0], spec[1])
		c.panel.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
		cards.append(c)
		stack_box.add_child(c.panel)
	cards[2].panel.add_theme_stylebox_override("panel", _style(Color(0.03, 0.03, 0.06, 0.95), ORANGE, 2, 8))
	add_child(stack_box)

	# Right: jutsu and its seal order.
	right_box = VBoxContainer.new()
	right_box.add_theme_constant_override("separation", 8)
	right_box.custom_minimum_size = Vector2(360, 0)
	jutsu_lbl = _label(26, ORANGE, true)
	jutsu_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	right_box.add_child(jutsu_lbl)
	progress_bar = _bar(Vector2(360, 6), ORANGE)
	right_box.add_child(progress_bar)
	for i in 5:
		rows.append(_make_row(i))
		right_box.add_child(rows[i].panel)
	reroll_btn = _button(pick_jutsu, 16)
	reroll_btn.custom_minimum_size = Vector2(0, 40)
	right_box.add_child(reroll_btn)
	how_lbl = _label(13, DIM)
	how_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	how_lbl.custom_minimum_size = Vector2(360, 0)
	right_box.add_child(how_lbl)
	add_child(right_box)

	# Bottom: music HUD.
	hud = PanelContainer.new()
	hud.add_theme_stylebox_override("panel", _style(PANEL_BG, Color(ORANGE, 0.4), 1, 6))
	hud.custom_minimum_size = Vector2(560, 0)
	var hv := VBoxContainer.new()
	hv.add_theme_constant_override("separation", 6)
	var top := HBoxContainer.new()
	hud_bar_lbl = _label(16, GOLD, false, HORIZONTAL_ALIGNMENT_LEFT)
	hud_bar_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top.add_child(hud_bar_lbl)
	var dots := HBoxContainer.new()
	dots.add_theme_constant_override("separation", 10)
	for i in 4:
		var d := Panel.new()
		d.custom_minimum_size = Vector2(16, 16)
		d.add_theme_stylebox_override("panel", _style(Color("#2a1a0a"), Color("#5a3a1a"), 1, 8))
		beat_dots.append(d)
		dots.add_child(d)
	top.add_child(dots)
	hud_score_lbl = _label(16, GOLD, false, HORIZONTAL_ALIGNMENT_RIGHT)
	hud_score_lbl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top.add_child(hud_score_lbl)
	hv.add_child(top)
	hud_progress = _bar(Vector2(0, 12), ORANGE)
	hv.add_child(hud_progress)
	hud_phase = _label(18, Color("#ff9944"), true)
	hv.add_child(hud_phase)
	hud.add_child(hv)
	add_child(hud)

	fx_layer = Control.new()
	fx_layer.set_anchors_preset(Control.PRESET_FULL_RECT)
	fx_layer.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(fx_layer)

	flash = ColorRect.new()
	flash.color = Color(1, 0, 0, 0)
	flash.set_anchors_preset(Control.PRESET_FULL_RECT)
	flash.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(flash)

	# Jutsu result overlay.
	overlay = Control.new()
	overlay.set_anchors_preset(Control.PRESET_FULL_RECT)
	overlay.visible = false
	var dim := ColorRect.new()
	dim.color = Color(0, 0, 0, 0.72)
	dim.set_anchors_preset(Control.PRESET_FULL_RECT)
	overlay.add_child(dim)
	jutsu_particles = _particles(Color.WHITE, false)
	overlay.add_child(jutsu_particles)
	var ov := VBoxContainer.new()
	ov.set_anchors_preset(Control.PRESET_FULL_RECT)
	ov.alignment = BoxContainer.ALIGNMENT_CENTER
	ov.add_theme_constant_override("separation", 24)
	overlay_title = _label(76, Color.WHITE, true)
	overlay_title.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	overlay_title.pivot_offset = Vector2(800, 50)
	ov.add_child(overlay_title)
	overlay_sub = _label(26, GOLD)
	ov.add_child(overlay_sub)
	overlay_btns = HBoxContainer.new()
	overlay_btns.alignment = BoxContainer.ALIGNMENT_CENTER
	overlay_btns.add_theme_constant_override("separation", 12)
	for spec in [["again", _on_again], ["pick_song", _on_pick_song], ["go_practice", _on_go.bind("practice")], ["go_challenge", _on_go.bind("challenge")]]:
		var b := _button(spec[1], 18)
		b.custom_minimum_size = Vector2(150, 46)
		b.set_meta("key", spec[0])
		overlay_btns.add_child(b)
	ov.add_child(overlay_btns)
	overlay.add_child(ov)
	add_child(overlay)

	countdown_lbl = _label(180, ORANGE, true)
	_glow(countdown_lbl, ORANGE, 40)
	countdown_lbl.set_anchors_preset(Control.PRESET_FULL_RECT)
	countdown_lbl.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	countdown_lbl.visible = false
	add_child(countdown_lbl)

	# Seal guide (click a seal in the list).
	modal = Control.new()
	modal.set_anchors_preset(Control.PRESET_FULL_RECT)
	modal.visible = false
	var mdim := ColorRect.new()
	mdim.color = Color(0, 0, 0, 0.75)
	mdim.set_anchors_preset(Control.PRESET_FULL_RECT)
	mdim.gui_input.connect(func(e: InputEvent) -> void:
		if e is InputEventMouseButton and e.pressed:
			modal.visible = false
	)
	modal.add_child(mdim)
	var mc := CenterContainer.new()
	mc.set_anchors_preset(Control.PRESET_FULL_RECT)
	mc.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var mp := PanelContainer.new()
	mp.add_theme_stylebox_override("panel", _style(Color(0.04, 0.04, 0.07, 0.98), ORANGE, 2, 8))
	var mv := VBoxContainer.new()
	mv.add_theme_constant_override("separation", 12)
	var mh := HBoxContainer.new()
	mh.add_theme_constant_override("separation", 16)
	modal_kanji = _label(64, ORANGE, true)
	_glow(modal_kanji, ORANGE)
	mh.add_child(modal_kanji)
	modal_name = _label(22, GOLD, false, HORIZONTAL_ALIGNMENT_LEFT)
	modal_name.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	mh.add_child(modal_name)
	mv.add_child(mh)
	modal_tex = TextureRect.new()
	modal_tex.custom_minimum_size = Vector2(400, 400)
	modal_tex.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	modal_tex.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	mv.add_child(modal_tex)
	modal_close = _button(func() -> void: modal.visible = false, 18)
	mv.add_child(modal_close)
	mp.add_child(mv)
	mc.add_child(mp)
	modal.add_child(mc)
	add_child(modal)


func _bar(min_size: Vector2, color: Color) -> ProgressBar:
	var p := ProgressBar.new()
	p.custom_minimum_size = min_size
	p.show_percentage = false
	p.max_value = 100
	p.add_theme_stylebox_override("background", _style(Color("#1a1208"), Color("#3a2a1a"), 1, 3))
	var fill := _style(color, color, 0, 3)
	fill.content_margin_top = 0
	fill.content_margin_bottom = 0
	p.add_theme_stylebox_override("fill", fill)
	p.mouse_filter = Control.MOUSE_FILTER_IGNORE
	return p


func _make_row(i: int) -> Dictionary:
	var p := PanelContainer.new()
	p.custom_minimum_size = Vector2(360, 62)
	p.mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND
	p.gui_input.connect(func(e: InputEvent) -> void:
		if e is InputEventMouseButton and e.pressed and e.button_index == MOUSE_BUTTON_LEFT:
			_open_modal(i)
	)
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 12)
	h.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var num := _label(14, DIM)
	num.custom_minimum_size = Vector2(18, 0)
	num.text = str(i + 1)
	h.add_child(num)
	var kanji := _label(34, GOLD, true)
	kanji.custom_minimum_size = Vector2(44, 0)
	h.add_child(kanji)
	var info := VBoxContainer.new()
	info.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	info.add_theme_constant_override("separation", 0)
	info.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var name := _label(18, GOLD, false, HORIZONTAL_ALIGNMENT_LEFT)
	var hint := _label(13, DIM, false, HORIZONTAL_ALIGNMENT_LEFT)
	info.add_child(name)
	info.add_child(hint)
	h.add_child(info)
	var check := _label(26, GREEN, true)
	check.custom_minimum_size = Vector2(28, 0)
	h.add_child(check)
	p.add_child(h)
	return {"panel": p, "kanji": kanji, "name": name, "hint": hint, "check": check, "num": num}


func _particles(color: Color, one_shot: bool) -> CPUParticles2D:
	var p := CPUParticles2D.new()
	p.emitting = false
	p.one_shot = one_shot
	p.amount = 40 if one_shot else 160
	p.lifetime = 1.0 if one_shot else 1.6
	p.explosiveness = 0.9 if one_shot else 0.0
	p.direction = Vector2(0, -1)
	p.spread = 180
	p.initial_velocity_min = 120
	p.initial_velocity_max = 380 if one_shot else 520
	p.gravity = Vector2(0, 260)
	p.scale_amount_min = 3
	p.scale_amount_max = 8
	p.color = color
	var ramp := Gradient.new()
	ramp.set_color(0, Color(1, 1, 1, 1))
	ramp.set_color(1, Color(1, 1, 1, 0))
	p.color_ramp = ramp
	if not one_shot:
		p.emission_shape = CPUParticles2D.EMISSION_SHAPE_SPHERE
		p.emission_sphere_radius = 60
	return p


func _layout() -> void:
	var vs := get_viewport_rect().size
	for c in [title_lbl, subtitle_lbl, mode_box, song_box, timer_box, left_box, stack_box, right_box, hud, lang_btn]:
		c.reset_size()
	title_lbl.position = Vector2((vs.x - title_lbl.size.x) / 2, 14)
	subtitle_lbl.position = Vector2((vs.x - subtitle_lbl.size.x) / 2, 76)
	lang_btn.position = Vector2(vs.x - lang_btn.size.x - 24, 20)
	mode_box.position = Vector2((vs.x - mode_box.size.x) / 2, 104)
	song_box.position = Vector2((vs.x - song_box.size.x) / 2, 160)
	timer_box.position = Vector2((vs.x - timer_box.size.x) / 2, 162)
	left_box.position = Vector2(24, (vs.y - left_box.size.y) / 2 + 40)
	right_box.position = Vector2(vs.x - right_box.size.x - 24, (vs.y - right_box.size.y) / 2 + 40)
	var top := 210.0
	var bottom: float = vs.y - (hud.size.y + 30 if hud.visible else 20.0)
	stack_box.position = Vector2((vs.x - stack_box.size.x) / 2, top + max(0.0, (bottom - top - stack_box.size.y) / 2))
	hud.position = Vector2((vs.x - hud.size.x) / 2, vs.y - hud.size.y - 18)
	jutsu_particles.position = vs / 2
	overlay_title.pivot_offset = overlay_title.size / 2


# ───────────────────────── text ─────────────────────────

func t(key: String) -> String:
	return Data.TEXT[key][lang]


func _toggle_lang() -> void:
	lang = 1 - lang
	_apply_lang()


func _apply_lang() -> void:
	title_lbl.text = t("title")
	subtitle_lbl.text = t("subtitle")
	lang_btn.text = t("lang")
	for m in mode_btns:
		mode_btns[m].text = t(m)
	timer_lbl.text = t("timer")
	reroll_btn.text = t("reroll")
	how_lbl.text = t("how")
	modal_close.text = t("close")
	start_btn.text = t("retry") if vision.phase == "error" else t("start")
	for b in overlay_btns.get_children():
		b.text = t(b.get_meta("key"))
	if gesture_lbl.text == "" or gesture_lbl.text.begins_with("—"):
		gesture_lbl.text = t("none")
	_update_hud_labels()
	if not jutsu.is_empty():
		_render()
	_layout()


func _update_hud_labels() -> void:
	hud_bar_lbl.text = "%s %d" % [t("bar"), m_bar_num]
	hud_score_lbl.text = "%s %d" % [t("score"), m_score]


# ───────────────────────── frame loop ─────────────────────────

func _process(delta: float) -> void:
	_update_status()
	var cam_visible := running and not debug and not overlay.visible and not modal.visible and not countdown_lbl.visible
	vision.set_cam_rect(cam_frame.get_global_rect().grow(-2), get_viewport_rect().size, cam_visible)

	if ch_active and not celebrating:
		ch_left -= delta
		timer_bar.value = max(ch_left, 0.0) / CHALLENGE_SECONDS * 100
		var urgent := ch_left <= 3.0
		(timer_bar.get_theme_stylebox("fill") as StyleBoxFlat).bg_color = RED if urgent else ORANGE
		if ch_left <= 0:
			ch_active = false
			_on_challenge_timeout()

	if m_seal_active and not m_round_pending:
		m_seal_t += delta
		var frac: float = min(m_seal_t / m_bar_sec, 1.0)
		hud_progress.value = (1.0 - frac) * 100
		_set_beat(min(int(frac * 4), 3))
		if m_seal_t >= m_bar_sec:
			m_seal_active = false
			_on_seal_bar_end()


func _update_status() -> void:
	if debug:
		status_lbl.text = t("st_debug")
		cam_hint.text = ""
		return
	if not vision.available:
		status_lbl.text = t("st_web_only")
		return
	match vision.phase:
		"idle":
			status_lbl.text = t("st_idle")
		"camera":
			status_lbl.text = t("st_camera")
		"loading":
			status_lbl.text = t("st_loading") % int(vision.progress * 100)
		"ready":
			status_lbl.text = t("st_ready_ms") % vision.ms if vision.ms > 0 else t("st_ready")
			if not running:
				_on_vision_ready()
		"error":
			status_lbl.text = t("err_" + vision.error) if Data.TEXT.has("err_" + vision.error) else t("err_load-failed")
			if start_btn.disabled:
				start_btn.disabled = false
				start_btn.text = t("retry")
	cam_hint.text = "" if running else "◉"


func _on_start() -> void:
	if debug:
		_on_vision_ready()
		return
	if not vision.available:
		return
	start_btn.disabled = true
	vision.start()


func _on_vision_ready() -> void:
	running = true
	start_btn.visible = false
	_layout()
	if mode == "challenge":
		_start_challenge_timer()
	elif mode == "music":
		start_music()


func _unhandled_input(event: InputEvent) -> void:
	if not debug or not (event is InputEventKey) or not event.pressed or event.echo:
		return
	var keys := [KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9, KEY_0, KEY_MINUS, KEY_EQUAL]
	var i := keys.find(event.keycode)
	var kanji := ""
	if event.keycode == KEY_SPACE:  # space = the seal that is wanted now
		if jutsu.is_empty() or step >= jutsu.seals.size():
			return
		kanji = jutsu.seals[step]
	elif i >= 0:
		kanji = Data.SEALS[i].kanji
	else:
		return
	var model_name: String = Data.GESTURE_MAP.find_key(kanji)
	for n in HOLD_FRAMES:
		_on_prediction(model_name, 0.95)


# ───────────────────────── modes ─────────────────────────

func set_mode(m: String) -> void:
	gen += 1
	ch_active = false
	if m_playing or m_starting or mode == "music":
		stop_music()
	celebrating = false
	resetting = false
	m_round_pending = false
	overlay.visible = false
	countdown_lbl.visible = false
	_stop_jutsu_particles()
	mode = m
	for k in mode_btns:
		_set_active(mode_btns[k], k == m)
	timer_box.visible = m == "challenge"
	song_box.visible = m == "music"
	hud.visible = m == "music"
	reroll_btn.visible = m != "music"
	_update_song_buttons()
	if jutsu.is_empty() or step >= jutsu.seals.size():
		pick_jutsu()
	else:
		_render()
	if m == "challenge":
		timer_bar.value = 100
		if running:
			_start_challenge_timer()
	elif m == "music":
		hud_phase.text = t("waiting") if not running else ""
		if running:
			start_music()
	_layout()


func pick_jutsu() -> void:
	if celebrating:
		return
	jutsu_idx = randi() % Data.JUTSUS.size()
	jutsu = Data.JUTSUS[jutsu_idx]
	step = 0
	hold = 0
	last_seen = ""
	_render()


func _render() -> void:
	jutsu_lbl.text = jutsu[["zh", "en"][lang]]
	jutsu_lbl.add_theme_color_override("font_color", jutsu.color)
	_glow(jutsu_lbl, jutsu.color, 16)
	for i in 5:
		var seal: Dictionary = _seal(jutsu.seals[i])
		rows[i].kanji.text = seal.kanji
		rows[i].name.text = seal.romaji
		rows[i].hint.text = seal.zh if lang == 0 else "%s seal" % seal.en
	_update_rows()
	_update_cards()


func _seal(kanji: String) -> Dictionary:
	for s in Data.SEALS:
		if s.kanji == kanji:
			return s
	return {}


func _update_rows() -> void:
	progress_bar.value = float(step) / jutsu.seals.size() * 100
	for i in 5:
		var r: Dictionary = rows[i]
		var state := "pending"
		if i < step:
			state = "done"
			if mode == "music" and i < m_results.size() and not m_results[i]:
				state = "miss"
		elif i == step and not celebrating:
			state = "current"
		var border: Color = {"pending": LINE, "current": ORANGE, "done": GREEN, "miss": RED}[state]
		var bg: Color = {"pending": PANEL_BG, "current": Color(0.12, 0.06, 0.01, 0.92), "done": Color(0.03, 0.09, 0.03, 0.9), "miss": Color(0.09, 0.03, 0.03, 0.9)}[state]
		r.panel.add_theme_stylebox_override("panel", _style(bg, border, 2 if state == "current" else 1))
		r.panel.modulate.a = 0.55 if state == "pending" else 1.0
		r.check.text = "✓" if state == "done" else ("×" if state == "miss" else "")
		r.check.add_theme_color_override("font_color", RED if state == "miss" else GREEN)


func _update_cards() -> void:
	var n: int = jutsu.seals.size()
	for slot in 3:
		var idx := step + (2 - slot)
		var c: Dictionary = cards[slot]
		var show := idx < n
		c.tex.texture = _seal_tex[jutsu.seals[idx]] if show else null
		c.panel.self_modulate.a = 1.0 if show else 0.0


func _pop_current_card() -> void:
	var p: Panel = cards[2].panel
	p.scale = Vector2(1.12, 1.12)
	create_tween().tween_property(p, "scale", Vector2.ONE, 0.35).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)


func _card_center() -> Vector2:
	return cards[2].panel.get_global_rect().get_center()


# ───────────────────────── recognition ─────────────────────────

func _on_prediction(gesture: String, conf: float) -> void:
	var mapped: String = Data.GESTURE_MAP.get(gesture, gesture)
	var seen := conf >= THRESHOLD and gesture != ""
	gesture_lbl.text = mapped if seen else t("none")
	conf_lbl.text = "%d%%" % int(conf * 100) if seen else ""
	if not running or celebrating:
		return
	if mode == "music":
		_music_gesture(mapped, seen)
		return
	if resetting or step >= jutsu.seals.size():
		return
	if seen and mapped == jutsu.seals[step]:
		hold = hold + 1 if mapped == last_seen else 1
		last_seen = mapped
		if hold >= HOLD_FRAMES:
			hold = 0
			last_seen = ""
			_on_seal_success()
	else:
		hold = 0
		last_seen = ""


func _on_seal_success() -> void:
	_spawn_ring(_card_center(), Color("#ff9944"))
	_burst(_card_center(), Color("#ff9944"))
	step += 1
	_update_rows()
	_update_cards()
	_pop_current_card()
	if step >= jutsu.seals.size():
		ch_active = false
		_later(0.7, _trigger_jutsu)
	elif mode == "challenge":
		_start_challenge_timer()


func _trigger_jutsu() -> void:
	celebrating = true
	_show_overlay(jutsu[["zh", "en"][lang]], jutsu.color, t("jutsu_done"), false)
	_start_jutsu_particles(jutsu.color)
	_play_voice(jutsu_idx + 1)
	_later(4.0, func() -> void:
		overlay.visible = false
		_stop_jutsu_particles()
		celebrating = false
		pick_jutsu()
		if mode == "challenge" and running:
			_start_challenge_timer()
	)


func _start_challenge_timer() -> void:
	ch_left = CHALLENGE_SECONDS
	timer_bar.value = 100
	ch_active = true


func _on_challenge_timeout() -> void:
	resetting = true
	cam_frame.add_theme_stylebox_override("panel", _style(Color(0.15, 0.0, 0.0, 0.95), Color.RED, 3))
	_flash_red()
	_later(0.6, func() -> void:
		cam_frame.add_theme_stylebox_override("panel", _style(Color(0.05, 0.05, 0.08, 0.95), Color(ORANGE, 0.4), 2))
		step = 0
		hold = 0
		last_seen = ""
		resetting = false
		_render()
		_start_challenge_timer()
	)


# ───────────────────────── music mode ─────────────────────────

func select_song(i: int) -> void:
	song_idx = i
	_update_song_buttons()
	if mode == "music" and running:
		gen += 1
		stop_music()
		start_music()


func _update_song_buttons() -> void:
	for i in song_btns.size():
		_set_active(song_btns[i], i == song_idx)


func start_music() -> void:
	if m_playing or m_starting:
		return
	m_starting = true
	var my_gen := gen
	var song: Dictionary = Data.SONGS[song_idx]
	m_bar_sec = 60.0 / song.bpm * 4.0
	m_rounds = 0
	m_score = 0
	m_streak = 0
	m_bar_num = 0
	m_seal_active = false
	m_seal_hit = false
	m_results = []
	m_round_pending = false
	celebrating = false
	overlay.visible = false
	hud.visible = true
	hud_progress.value = 0
	_set_beat(-1)
	_update_hud_labels()
	hud_phase.text = t("loading_song")
	_layout()
	var stream: AudioStream = await _load_audio("music/" + song.file, "music/" + song.src)
	if my_gen != gen:
		return
	if stream == null:
		m_starting = false
		hud_phase.text = t("song_failed")
		return
	music_player.stream = stream
	hud_phase.text = t("get_ready")
	_countdown(3, func() -> void:
		m_starting = false
		music_player.play()
		m_playing = true
		_begin_round()
	)


func stop_music() -> void:
	m_playing = false
	m_starting = false
	m_seal_active = false
	m_seal_hit = false
	m_results = []
	m_round_pending = false
	m_rounds = 0
	celebrating = false
	music_player.stop()
	countdown_lbl.visible = false
	overlay.visible = false
	_stop_jutsu_particles()
	hud_progress.value = 0
	_set_beat(-1)
	hud_phase.text = ""
	gesture_lbl.text = t("none")
	conf_lbl.text = ""
	step = 0
	hold = 0
	last_seen = ""
	jutsu = {}
	pick_jutsu()


func _countdown(n: int, done: Callable) -> void:
	if n == 0:
		countdown_lbl.visible = false
		done.call()
		return
	countdown_lbl.visible = true
	countdown_lbl.text = str(n)
	countdown_lbl.pivot_offset = countdown_lbl.size / 2
	countdown_lbl.scale = Vector2(1.8, 1.8)
	countdown_lbl.modulate.a = 0.0
	var tw := create_tween().set_parallel()
	tw.tween_property(countdown_lbl, "scale", Vector2.ONE, 0.5).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	tw.tween_property(countdown_lbl, "modulate:a", 1.0, 0.25)
	_later(0.9, _countdown.bind(n - 1, done))


func _begin_round() -> void:
	if m_rounds >= MUSIC_MAX_ROUNDS:
		_music_game_over()
		return
	m_results = []
	m_round_pending = false
	celebrating = false
	jutsu = {}
	pick_jutsu()
	hud_progress.value = 0
	_set_beat(-1)
	hud_phase.text = t("seal_n") % [1, jutsu.seals.size()]
	_start_seal_timer()


func _start_seal_timer() -> void:
	m_seal_t = 0.0
	m_seal_hit = false
	hold = 0
	last_seen = ""
	(hud_progress.get_theme_stylebox("fill") as StyleBoxFlat).bg_color = ORANGE
	m_seal_active = true


func _music_gesture(mapped: String, seen: bool) -> void:
	if not m_playing or not m_seal_active or m_seal_hit or m_round_pending or step >= jutsu.seals.size():
		return
	if seen and mapped == jutsu.seals[step]:
		hold = hold + 1 if mapped == last_seen else 1
		last_seen = mapped
		if hold >= HOLD_FRAMES:
			hold = 0
			last_seen = ""
			m_seal_hit = true
			(hud_progress.get_theme_stylebox("fill") as StyleBoxFlat).bg_color = Color("#66cc66")
			_spawn_ring(_card_center(), Color("#ff9944"))
			_burst(_card_center(), Color("#ff9944"))
			rows[step].panel.add_theme_stylebox_override("panel", _style(Color(0.03, 0.09, 0.03, 0.9), GREEN))
			rows[step].check.text = "✓"
			rows[step].check.add_theme_color_override("font_color", GREEN)
	else:
		hold = 0
		last_seen = ""


func _on_seal_bar_end() -> void:
	if not m_playing or m_round_pending:
		return
	m_results.append(m_seal_hit)
	if not m_seal_hit:
		_flash_red()
	step += 1
	_update_rows()
	_update_cards()
	if m_seal_hit:
		_pop_current_card()
	m_bar_num += 1
	_update_hud_labels()
	hud_progress.value = 0
	_set_beat(-1)
	if step >= jutsu.seals.size():
		m_round_pending = true
		celebrating = true
		_later(0.12, _on_round_end)
	else:
		hud_phase.text = t("seal_n") % [step + 1, jutsu.seals.size()]
		_later(0.15, _start_seal_timer)


func _on_round_end() -> void:
	var correct := m_results.count(true)
	var total := m_results.size()
	var all_good := correct >= total - 1
	m_rounds += 1
	m_score += correct
	_update_hud_labels()
	if all_good:
		m_streak += 1
		var sub := t("perfect") % [correct, total] if correct == total else t("good") % [correct, total]
		_show_overlay(jutsu[["zh", "en"][lang]], jutsu.color, sub, false)
		_start_jutsu_particles(jutsu.color)
		_play_voice(jutsu_idx + 1)
		if m_streak == 2:
			_later(0.8, _play_voice.bind(6))
		elif m_streak >= 3:
			_later(0.8, _play_voice.bind(7))
	else:
		m_streak = 0
		_show_overlay(jutsu[["zh", "en"][lang]], YELLOW, t("result") % [correct, total], false)
		if correct > 0:
			_burst(get_viewport_rect().size / 2, YELLOW)
	if m_rounds >= MUSIC_MAX_ROUNDS:
		_later(2.5, _music_game_over)
	else:
		_later(2.5, _next_round)


func _next_round() -> void:
	overlay.visible = false
	_stop_jutsu_particles()
	if not m_playing:
		return
	_begin_round()


func _music_game_over() -> void:
	m_playing = false
	m_seal_active = false
	m_round_pending = true
	celebrating = true
	music_player.stop()
	_stop_jutsu_particles()
	var sub := t("final") % [m_score, min(m_rounds, MUSIC_MAX_ROUNDS), MUSIC_MAX_ROUNDS]
	_show_overlay(t("game_over"), YELLOW, sub, true)


func _on_song_finished() -> void:
	if mode != "music":
		return
	m_playing = false
	m_seal_active = false
	_set_beat(-1)
	if not m_round_pending and not celebrating:
		hud_phase.text = t("song_over")


func _on_again() -> void:
	overlay.visible = false
	gen += 1
	stop_music()
	start_music()


func _on_pick_song() -> void:
	overlay.visible = false
	gen += 1
	stop_music()
	hud_phase.text = t("waiting") if not running else "♪"


func _on_go(m: String) -> void:
	overlay.visible = false
	set_mode(m)


func _set_beat(active: int) -> void:
	for i in 4:
		var on := i == active
		beat_dots[i].add_theme_stylebox_override("panel", _style(ORANGE if on else Color("#2a1a0a"), Color("#ffaa55") if on else Color("#5a3a1a"), 1, 8))


# ───────────────────────── effects ─────────────────────────

func _show_overlay(title: String, color: Color, sub: String, buttons: bool) -> void:
	overlay_title.text = title
	overlay_title.add_theme_color_override("font_color", color)
	_glow(overlay_title, color, 40)
	overlay_sub.text = sub
	overlay_btns.visible = buttons
	overlay.visible = true
	overlay_title.pivot_offset = overlay_title.size / 2
	overlay_title.scale = Vector2(0.6, 0.6)
	overlay_title.modulate.a = 0.0
	var tw := create_tween().set_parallel()
	tw.tween_property(overlay_title, "scale", Vector2.ONE, 0.6).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	tw.tween_property(overlay_title, "modulate:a", 1.0, 0.3)


func _start_jutsu_particles(color: Color) -> void:
	jutsu_particles.position = get_viewport_rect().size / 2
	jutsu_particles.color = color
	jutsu_particles.emitting = true


func _stop_jutsu_particles() -> void:
	if jutsu_particles:
		jutsu_particles.emitting = false


func _burst(at: Vector2, color: Color) -> void:
	var p := _particles(color, true)
	p.position = at
	fx_layer.add_child(p)
	p.emitting = true
	p.finished.connect(p.queue_free)


func _spawn_ring(at: Vector2, color: Color) -> void:
	var r := Panel.new()
	var sb := StyleBoxFlat.new()
	sb.draw_center = false
	sb.border_color = color
	sb.set_border_width_all(3)
	sb.set_corner_radius_all(40)
	r.add_theme_stylebox_override("panel", sb)
	r.size = Vector2(80, 80)
	r.pivot_offset = Vector2(40, 40)
	r.position = at - Vector2(40, 40)
	r.mouse_filter = Control.MOUSE_FILTER_IGNORE
	fx_layer.add_child(r)
	var tw := create_tween().set_parallel()
	tw.tween_property(r, "scale", Vector2(4.5, 4.5), 0.5).set_ease(Tween.EASE_OUT)
	tw.tween_property(r, "modulate:a", 0.0, 0.5)
	tw.chain().tween_callback(r.queue_free)


func _flash_red() -> void:
	flash.color = Color(1, 0, 0, 0.27)
	create_tween().tween_property(flash, "color:a", 0.0, 0.4)


func _open_modal(i: int) -> void:
	if jutsu.is_empty():
		return
	var seal := _seal(jutsu.seals[i])
	modal_kanji.text = seal.kanji
	modal_name.text = "%s\n%s" % [seal.romaji, seal.zh if lang == 0 else seal.en + " seal"]
	modal_tex.texture = _seal_tex[seal.kanji]
	modal.visible = true


# ───────────────────────── sound + timing ─────────────────────────

## Run cb after sec seconds unless the mode changed in between.
func _later(sec: float, cb: Callable) -> void:
	var g := gen
	get_tree().create_timer(sec).timeout.connect(func() -> void:
		if g == gen:
			cb.call()
	)


func _play_voice(n: int) -> void:
	var stream: AudioStream = await _load_audio("audio/%d.mp3" % n, "audio/%d.mp3" % n)
	if stream == null:
		return
	var p := AudioStreamPlayer.new()
	p.stream = stream
	add_child(p)
	p.finished.connect(p.queue_free)
	p.play()


## web_path is relative to the page; repo_path is relative to the repo root
## (used when running from the editor).
func _load_audio(web_path: String, repo_path: String) -> AudioStream:
	if _audio_cache.has(web_path):
		var cached = _audio_cache[web_path]
		if cached is AudioStream:
			return cached
		await cached.loaded  # another call is downloading it
		return _audio_cache.get(web_path) as AudioStream
	var waiter := AudioWaiter.new()
	_audio_cache[web_path] = waiter
	var bytes := PackedByteArray()
	if OS.has_feature("web"):
		var req := HTTPRequest.new()
		add_child(req)
		if req.request(base_url + web_path.uri_encode().replace("%2F", "/")) == OK:
			var res: Array = await req.request_completed
			if res[0] == HTTPRequest.RESULT_SUCCESS and res[1] == 200:
				bytes = res[3]
		req.queue_free()
	else:
		bytes = FileAccess.get_file_as_bytes(ProjectSettings.globalize_path("res://").path_join("../../" + repo_path))
	var stream: AudioStream = null
	if bytes.size() > 0:
		var mp3 := AudioStreamMP3.new()
		mp3.data = bytes
		stream = mp3
		_audio_cache[web_path] = stream
	else:
		_audio_cache.erase(web_path)
	waiter.loaded.emit()
	return stream


class AudioWaiter:
	signal loaded
