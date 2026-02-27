shader_type spatial;
render_mode cull_disabled, shadows_disabled;

// Procedural grass blade shader with wind sway animation.
// Features: secondary flutter, curved blade tips, rim lighting, waxy sheen.
// Uses Y-axis billboard quads via MultiMeshInstance3D.

uniform vec3 grass_color_base : source_color = vec3(0.25, 0.40, 0.15);
uniform vec3 grass_color_tip : source_color = vec3(0.45, 0.58, 0.25);
uniform float wind_strength : hint_range(0.0, 2.0) = 0.4;
uniform float wind_speed : hint_range(0.0, 5.0) = 1.5;
uniform float wind_turbulence : hint_range(0.0, 2.0) = 0.3;
uniform float alpha_cutoff : hint_range(0.0, 1.0) = 0.3;
uniform vec3 player_world_pos = vec3(0.0, 0.0, 0.0);

void vertex() {
	// Wind displacement — stronger at top of blade (VERTEX.y > 0)
	float height_factor = clamp(VERTEX.y, 0.0, 1.0);
	vec3 world_pos = (MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

	float wind_phase = world_pos.x * 0.7 + world_pos.z * 0.5 + TIME * wind_speed;

	// Primary sway
	float wind = sin(wind_phase) * wind_strength * height_factor;
	float turbulence = sin(wind_phase * 3.7 + world_pos.x * 1.3) * wind_turbulence * height_factor * 0.3;

	// Secondary flutter — high-frequency tip vibration
	float flutter = sin(wind_phase * 5.2 + TIME * wind_speed * 2.5) * 0.05 * height_factor;

	// Curved blade tips — quadratic displacement so base stays planted
	float curve_factor = height_factor * height_factor;
	VERTEX.x += (wind + turbulence + flutter) * curve_factor;
	VERTEX.z += (wind * 0.3 + turbulence * 0.5) * curve_factor;

	// Player proximity push — blades part around the operator
	vec2 to_player = world_pos.xz - player_world_pos.xz;
	float dist = length(to_player);
	float push = smoothstep(1.5, 0.3, dist) * height_factor;
	if (dist > 0.01) {
		vec2 push_dir = normalize(to_player) * push * 0.4;
		VERTEX.xz += push_dir;
		VERTEX.y -= push * 0.15; // flatten toward ground
	}
}

void fragment() {
	// Gradient from base to tip color based on UV.y
	vec3 color = mix(grass_color_base, grass_color_tip, UV.y);

	// Slight variation from facing angle
	float ndv = abs(dot(NORMAL, VIEW));
	color *= mix(0.85, 1.0, ndv);

	// Rim lighting — fake subsurface scattering on edges facing away from view
	float rim = 1.0 - ndv;
	color *= mix(1.0, 1.4, rim * 0.3);

	ALBEDO = color;
	ROUGHNESS = 0.65; // Waxy cuticle sheen
	METALLIC = 0.0;

	// Alpha: fade out edges of the quad for soft grass blades
	float blade_shape = 1.0 - abs(UV.x * 2.0 - 1.0);
	blade_shape *= smoothstep(0.0, 0.1, UV.y) * smoothstep(1.0, 0.7, UV.y);
	ALPHA = blade_shape > alpha_cutoff ? 1.0 : 0.0;
	ALPHA_SCISSOR_THRESHOLD = alpha_cutoff;
}
