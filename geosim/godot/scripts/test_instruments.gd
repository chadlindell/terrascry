## Headless instrument test — exercises query path for all 4 instruments.
## Run: godot --headless --path godot/ res://scenes/test_instruments.tscn
extends Node

var _done := false


func _ready() -> void:
	print("\n=== INSTRUMENT MOCK TEST ===\n")
	await _run_tests()
	_done = true


func _process(_delta: float) -> void:
	if _done:
		get_tree().quit()


func _sci(val: float) -> String:
	## Format float in scientific-like notation (GDScript lacks %e).
	if val == 0.0:
		return "0.0"
	var sign_str := ""
	if val < 0:
		sign_str = "-"
		val = -val
	var exp_val := int(floor(log(val) / log(10.0)))
	var mantissa := val / pow(10.0, exp_val)
	return "%s%.3f e%d" % [sign_str, mantissa, exp_val]


func _run_tests() -> void:
	print("  PhysicsClient.is_mock_mode = %s" % PhysicsClient.is_mock_mode)

	# --- Step 1: Load scenario ---
	print("\n[1] Loading scenario...")
	var load_result = await PhysicsClient.load_scenario("scenarios/swamp-crash-site.json")
	print("  load_scenario status: %s" % load_result.get("status"))

	var info_result = await PhysicsClient.get_scenario_info()
	print("  get_scenario_info status: %s" % info_result.get("status"))
	var info: Dictionary = info_result.get("data", {})
	var objects: Array = info.get("objects", [])
	print("  n_objects: %d" % objects.size())
	if objects.size() > 0:
		var obj0: Dictionary = objects[0]
		print("  objects[0]: name=%s pos=%s radius=%s" % [
			obj0.get("name", "?"), obj0.get("position", []),
			obj0.get("radius", 0)])
		print("  objects[0] has susceptibility: %s" % obj0.has("susceptibility"))
		print("  objects[0] has conductivity: %s" % obj0.has("conductivity"))
	var avail: Array = info.get("available_instruments", [])
	print("  available_instruments: %s" % [avail])

	# Store in SurveyManager like the real flow does
	SurveyManager.scenario_info = info

	# Pick a target to test near
	var target_pos := [10.0, 10.0, -1.0]
	if objects.size() > 0:
		target_pos = objects[0].get("position", target_pos)
	print("  Testing near target at: %s" % [target_pos])
	var near_pos := [target_pos[0], target_pos[1], 0.175]
	var far_pos := [2.0, 2.0, 0.175]
	var md_near := [target_pos[0], target_pos[1], 0.04]
	var md_far := [2.0, 2.0, 0.04]

	# --- Step 2: MAG_GRADIOMETER ---
	print("\n[2] MAG_GRADIOMETER ---")
	SurveyManager.current_instrument = SurveyManager.Instrument.MAG_GRADIOMETER
	var grad_near = await PhysicsClient.query_gradient([near_pos])
	var grad_far = await PhysicsClient.query_gradient([far_pos])
	var gn_data: Dictionary = grad_near.get("data", {})
	var gf_data: Dictionary = grad_far.get("data", {})
	var gn: Array = gn_data.get("gradient", [0.0])
	var gf: Array = gf_data.get("gradient", [0.0])
	print("  gradient NEAR: %s T/m = %.1f nT/m" % [_sci(gn[0]), gn[0] * 1e9])
	print("  gradient FAR:  %s T/m = %.1f nT/m" % [_sci(gf[0]), gf[0] * 1e9])
	var pc: Array = gn_data.get("per_channel", [])
	print("  per_channel present: %s (size=%d)" % [not pc.is_empty(), pc.size()])
	if pc.size() > 0:
		var channels: Array = pc[0]
		print("  per_channel[0] (nT/m): %s" % [_arr_nT(channels)])
	var adc: Array = gn_data.get("adc_counts", [])
	if adc.size() > 0:
		print("  adc_counts[0]: %s" % [adc[0]])

	# --- Step 3: METAL_DETECTOR ---
	print("\n[3] METAL_DETECTOR ---")
	SurveyManager.current_instrument = SurveyManager.Instrument.METAL_DETECTOR
	var md_near_r = await PhysicsClient.query_metal_detector([md_near])
	var md_far_r = await PhysicsClient.query_metal_detector([md_far])
	print("  NEAR status: %s" % md_near_r.get("status"))
	var md_near_data: Dictionary = md_near_r.get("data", {})
	var md_far_data: Dictionary = md_far_r.get("data", {})
	var dt_near: Array = md_near_data.get("delta_t", [0.0])
	var dt_far: Array = md_far_data.get("delta_t", [0.0])
	print("  delta_t NEAR: %s T = %.1f nT" % [_sci(dt_near[0]), dt_near[0] * 1e9])
	print("  delta_t FAR:  %s T = %.1f nT" % [_sci(dt_far[0]), dt_far[0] * 1e9])
	var tid: Array = md_near_data.get("target_id", [-1])
	var depth: Array = md_near_data.get("depth_estimate", [-1.0])
	var ferrous: Array = md_near_data.get("ferrous_ratio", [-1.0])
	print("  target_id: %s  depth: %.2f m  ferrous: %.2f" % [tid[0], depth[0], ferrous[0]])
	if md_near_r.get("message", "").contains("mock"):
		print("  ** Used MOCK fallback (server lacks query_metal_detector)")

	# --- Step 4: EM_FDEM ---
	print("\n[4] EM_FDEM ---")
	SurveyManager.current_instrument = SurveyManager.Instrument.EM_FDEM
	var em_near_r = await PhysicsClient.query_em_response([near_pos], 10000.0)
	var em_far_r = await PhysicsClient.query_em_response([far_pos], 10000.0)
	var em_near_data: Dictionary = em_near_r.get("data", {})
	var em_far_data: Dictionary = em_far_r.get("data", {})
	var rr_near: Array = em_near_data.get("response_real", [0.0])
	var ri_near: Array = em_near_data.get("response_imag", [0.0])
	var rr_far: Array = em_far_data.get("response_real", [0.0])
	print("  response_real NEAR: %s = %.1f ppm" % [_sci(rr_near[0]), rr_near[0] * 1e6])
	print("  response_imag NEAR: %s = %.1f ppm" % [_sci(ri_near[0]), ri_near[0] * 1e6])
	print("  response_real FAR:  %s = %.1f ppm" % [_sci(rr_far[0]), rr_far[0] * 1e6])

	# --- Step 5: RESISTIVITY ---
	print("\n[5] RESISTIVITY ---")
	SurveyManager.current_instrument = SurveyManager.Instrument.RESISTIVITY
	var near_electrodes := [
		[target_pos[0] - 0.75, target_pos[1], 0.0],
		[target_pos[0] - 0.25, target_pos[1], 0.0],
		[target_pos[0] + 0.25, target_pos[1], 0.0],
		[target_pos[0] + 0.75, target_pos[1], 0.0],
	]
	var far_electrodes := [
		[1.25, 2.0, 0.0], [1.75, 2.0, 0.0],
		[2.25, 2.0, 0.0], [2.75, 2.0, 0.0],
	]
	var ert_near_r = await PhysicsClient.query_apparent_resistivity(near_electrodes, [[0, 1, 2, 3]])
	var ert_far_r = await PhysicsClient.query_apparent_resistivity(far_electrodes, [[0, 1, 2, 3]])
	print("  NEAR status: %s" % ert_near_r.get("status"))
	var rho_near: Array = ert_near_r.get("data", {}).get("apparent_resistivity", [0.0])
	var rho_far: Array = ert_far_r.get("data", {}).get("apparent_resistivity", [0.0])
	print("  rho NEAR: %.1f Ohm-m" % rho_near[0])
	print("  rho FAR:  %.1f Ohm-m" % rho_far[0])
	var ert_diff := absf(rho_near[0] - rho_far[0])
	print("  ERT near/far diff: %.2f Ohm-m  %s" % [
		ert_diff, "OK (spatial variation)" if ert_diff > 0.5 else "** FAIL: no variation **"])

	# --- Step 5b: SENSITIVITY CHECK ---
	print("\n[5b] SENSITIVITY CHECK ---")
	# Metal detector: dog tag at 1m depth should NOT saturate at 2m lateral distance
	var md_2m_pos := [target_pos[0] + 2.0, target_pos[1], 0.04]
	var md_2m_r = await PhysicsClient.query_metal_detector([md_2m_pos])
	var dt_2m: Array = md_2m_r.get("data", {}).get("delta_t", [0.0])
	var nT_2m := absf(dt_2m[0]) * 1e9
	print("  MD at 2m lateral: %.1f nT (should be < 50 nT for small target)" % nT_2m)
	# VCO full-scale is 100 nT, so < 50 nT means not saturated
	print("  VCO would be at %.0f%% (ref 100nT)" % [minf(nT_2m / 100.0, 1.0) * 100.0])

	# --- Step 6: HUD display simulation ---
	print("\n[6] HUD DISPLAY VALUES ---")
	_simulate_hud("MAG_GRADIOMETER", SurveyManager.Instrument.MAG_GRADIOMETER, gn[0])
	_simulate_hud("METAL_DETECTOR", SurveyManager.Instrument.METAL_DETECTOR, dt_near[0])
	_simulate_hud("EM_FDEM", SurveyManager.Instrument.EM_FDEM, rr_near[0])
	_simulate_hud("RESISTIVITY", SurveyManager.Instrument.RESISTIVITY, rho_near[0])

	# --- Step 7: set_environment ---
	print("\n[7] set_environment ---")
	var env_r = await PhysicsClient.set_environment({"moisture": 0.3})
	print("  status: %s" % env_r.get("status"))

	print("\n=== ALL TESTS COMPLETE ===\n")


func _simulate_hud(inst_name: String, inst: SurveyManager.Instrument, raw: float) -> void:
	SurveyManager.current_instrument = inst
	var units := SurveyManager.instrument_units(inst)
	var display_value: float
	match inst:
		SurveyManager.Instrument.MAG_GRADIOMETER, SurveyManager.Instrument.METAL_DETECTOR:
			display_value = raw * 1e9
		SurveyManager.Instrument.EM_FDEM:
			display_value = raw * 1e6
		SurveyManager.Instrument.RESISTIVITY:
			display_value = raw
		_:
			display_value = raw * 1e9
	print("  %s: raw=%s -> label=\"%.1f %s\"" % [inst_name, _sci(raw), display_value, units])


func _arr_nT(arr: Array) -> String:
	var parts := PackedStringArray()
	for v in arr:
		parts.append("%.1f" % [float(v) * 1e9])
	return "[%s]" % ", ".join(parts)
