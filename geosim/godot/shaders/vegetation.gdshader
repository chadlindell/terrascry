shader_type spatial;
render_mode cull_disabled;

// Vegetation shader for trees, bushes, and rocks.
// Per-instance custom data: x=hue_shift, y=wind_phase, z=ao_factor, w=unused.
// Features: vertical color gradient, hierarchical wind sway, ground-contact AO,
// hue variation, SSS backlight.

uniform vec3 base_color : source_color = vec3(0.18, 0.35, 0.12);
uniform vec3 tip_color : source_color = vec3(0.30, 0.50, 0.20);
uniform float wind_strength : hint_range(0.0, 1.0) = 0.15;
uniform float wind_speed : hint_range(0.0, 3.0) = 1.0;
uniform float gradient_height : hint_range(0.1, 5.0) = 2.0;
uniform float ao_strength : hint_range(0.0, 1.0) = 0.5;
uniform vec3 player_world_pos = vec3(0.0, 0.0, 0.0);
uniform bool use_vertex_height = false;
uniform sampler2D model_albedo : source_color, filter_linear_mipmap;
uniform bool has_model_texture = false;

// SSS backlight
uniform vec3 sss_color : source_color = vec3(0.35, 0.5, 0.2);
uniform float sss_strength : hint_range(0.0, 1.0) = 0.4;

// Pass per-instance data from vertex to fragment via varyings
varying float v_hue_shift;
varying float v_ao_factor;
varying float v_height_factor;

void vertex() {
	// Forward INSTANCE_CUSTOM to fragment shader
	v_hue_shift = INSTANCE_CUSTOM.x;
	v_ao_factor = INSTANCE_CUSTOM.z;

	float phase_offset = INSTANCE_CUSTOM.y * 6.283;
	float height_factor = clamp(VERTEX.y / gradient_height, 0.0, 1.0);
	v_height_factor = height_factor;
	float wind_phase = phase_offset + TIME * wind_speed;

	// Hierarchical wind: trunk sway (slow) + branch sway (medium) + leaf flutter (fast)
	// Layer 1: Trunk sway — slow, large-scale, affects entire tree
	float trunk_sway = sin(wind_phase * 0.4) * wind_strength * 0.3;
	// Layer 2: Branch sway — medium frequency, mid-height and above
	float branch_factor = smoothstep(0.3, 0.7, height_factor);
	float branch_sway = sin(wind_phase * 1.2 + 2.1) * wind_strength * 0.5 * branch_factor;
	// Layer 3: Leaf flutter — high frequency, tips only
	float leaf_factor = smoothstep(0.6, 1.0, height_factor);
	float leaf_flutter = sin(wind_phase * 3.5 + VERTEX.x * 4.0) * wind_strength * 0.25 * leaf_factor;
	float leaf_flutter_z = sin(wind_phase * 4.1 + VERTEX.z * 3.5 + 1.7) * wind_strength * 0.2 * leaf_factor;

	// Combined displacement with quadratic height gating
	float h2 = height_factor * height_factor;
	VERTEX.x += (trunk_sway + branch_sway + leaf_flutter) * h2;
	VERTEX.z += (trunk_sway * 0.4 + branch_sway * 0.6 + leaf_flutter_z) * h2;

	// Player proximity push — bushes sway away, trees unaffected (height gating)
	vec3 instance_pos = (MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	vec2 to_player = instance_pos.xz - player_world_pos.xz;
	float dist = length(to_player);
	float push = smoothstep(1.5, 0.3, dist) * height_factor * (1.0 - height_factor * 0.5);
	if (dist > 0.01) {
		vec2 push_dir = normalize(to_player) * push * 0.2;
		VERTEX.xz += push_dir;
	}
}

void fragment() {
	float height_factor = use_vertex_height ? v_height_factor : clamp(UV.y, 0.0, 1.0);

	// Hue variation from vertex varying
	float hue_shift = v_hue_shift;
	vec3 color_base = base_color;
	vec3 color_top = tip_color;
	color_base.r += hue_shift * 0.08;
	color_base.g -= abs(hue_shift) * 0.03;
	color_top.r += hue_shift * 0.06;
	color_top.g -= abs(hue_shift) * 0.02;

	vec3 color = mix(color_base, color_top, height_factor);

	if (has_model_texture) {
		vec3 tex_color = texture(model_albedo, UV).rgb;
		color = tex_color * (color / max(mix(base_color, tip_color, 0.5), vec3(0.01)));
	}

	// Ground-contact AO darkening
	float ao_factor = v_ao_factor;
	float ao = mix(1.0 - ao_strength, 1.0, smoothstep(0.0, 0.3, height_factor));
	ao *= mix(0.7, 1.0, ao_factor);
	color *= ao;

	ALBEDO = color;
	ROUGHNESS = 0.85;
	METALLIC = 0.0;

	// SSS backlight — leaves glow when sun is behind them
	float backlight_ndv = max(dot(-NORMAL, VIEW), 0.0);
	float sss_factor = backlight_ndv * sss_strength * smoothstep(0.2, 0.8, height_factor);
	BACKLIGHT = sss_color * sss_factor;
}
