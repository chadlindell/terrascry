shader_type spatial;
render_mode cull_disabled;

// Vegetation shader for trees, bushes, and rocks.
// Per-instance custom data: x=hue_shift, y=wind_phase, z=ao_factor, w=unused.
// Features: vertical color gradient, wind sway, ground-contact AO, hue variation.

uniform vec3 base_color : source_color = vec3(0.18, 0.35, 0.12);
uniform vec3 tip_color : source_color = vec3(0.30, 0.50, 0.20);
uniform float wind_strength : hint_range(0.0, 1.0) = 0.15;
uniform float wind_speed : hint_range(0.0, 3.0) = 1.0;
uniform float gradient_height : hint_range(0.1, 5.0) = 2.0;
uniform float ao_strength : hint_range(0.0, 1.0) = 0.5;
uniform vec3 player_world_pos = vec3(0.0, 0.0, 0.0);

// Pass per-instance data from vertex to fragment via varyings
varying float v_hue_shift;
varying float v_ao_factor;

void vertex() {
	// Forward INSTANCE_CUSTOM to fragment shader
	v_hue_shift = INSTANCE_CUSTOM.x;
	v_ao_factor = INSTANCE_CUSTOM.z;

	// Wind sway — modulated by per-instance wind phase (INSTANCE_CUSTOM.y)
	float phase_offset = INSTANCE_CUSTOM.y * 6.283;
	float height_factor = clamp(VERTEX.y / gradient_height, 0.0, 1.0);
	float wind_phase = phase_offset + TIME * wind_speed;
	float sway = sin(wind_phase) * wind_strength * height_factor * height_factor;
	float sway_z = sin(wind_phase * 0.7 + 1.5) * wind_strength * 0.5 * height_factor * height_factor;
	VERTEX.x += sway;
	VERTEX.z += sway_z;

	// Player proximity push — bushes sway away, trees unaffected (height gating)
	vec3 instance_pos = (MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	vec2 to_player = instance_pos.xz - player_world_pos.xz;
	float dist = length(to_player);
	// Weaker effect than grass; height_factor gates so tall trees barely move
	float push = smoothstep(1.5, 0.3, dist) * height_factor * (1.0 - height_factor * 0.5);
	if (dist > 0.01) {
		vec2 push_dir = normalize(to_player) * push * 0.2;
		VERTEX.xz += push_dir;
	}
}

void fragment() {
	// Height-based gradient (model-space Y)
	float height_factor = clamp(UV.y, 0.0, 1.0);

	// Hue variation from vertex varying
	float hue_shift = v_hue_shift;
	vec3 color_base = base_color;
	vec3 color_top = tip_color;
	// Shift hue by rotating green/red channels
	color_base.r += hue_shift * 0.08;
	color_base.g -= abs(hue_shift) * 0.03;
	color_top.r += hue_shift * 0.06;
	color_top.g -= abs(hue_shift) * 0.02;

	vec3 color = mix(color_base, color_top, height_factor);

	// Ground-contact AO darkening
	float ao_factor = v_ao_factor;
	float ao = mix(1.0 - ao_strength, 1.0, smoothstep(0.0, 0.3, height_factor));
	ao *= mix(0.7, 1.0, ao_factor);
	color *= ao;

	ALBEDO = color;
	ROUGHNESS = 0.85;
	METALLIC = 0.0;
}
