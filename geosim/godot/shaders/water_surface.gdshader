shader_type spatial;
render_mode cull_disabled, depth_draw_always;

// Water surface shader with Gerstner waves, depth fade, foam, scrolling normals,
// fresnel reflection, color absorption. Designed for swamp/field survey environments.

// Water color
uniform vec3 shallow_color : source_color = vec3(0.15, 0.22, 0.18);
uniform vec3 deep_color : source_color = vec3(0.03, 0.06, 0.08);
uniform float color_depth_factor : hint_range(0.5, 10.0) = 3.0;

// Transparency
uniform float base_alpha : hint_range(0.0, 1.0) = 0.7;
uniform float depth_fade_distance : hint_range(0.1, 5.0) = 1.5;

// Fresnel
uniform float fresnel_power : hint_range(1.0, 10.0) = 4.0;
uniform float fresnel_bias : hint_range(0.0, 0.5) = 0.05;

// Gerstner wave parameters (3 wave layers)
uniform float wave_amplitude_1 : hint_range(0.0, 0.2) = 0.03;
uniform float wave_frequency_1 : hint_range(0.1, 5.0) = 1.2;
uniform float wave_speed_1 : hint_range(0.0, 3.0) = 0.8;
uniform vec2 wave_direction_1 = vec2(1.0, 0.3);

uniform float wave_amplitude_2 : hint_range(0.0, 0.2) = 0.02;
uniform float wave_frequency_2 : hint_range(0.1, 5.0) = 2.0;
uniform float wave_speed_2 : hint_range(0.0, 3.0) = 0.5;
uniform vec2 wave_direction_2 = vec2(0.4, 1.0);

uniform float wave_amplitude_3 : hint_range(0.0, 0.2) = 0.01;
uniform float wave_frequency_3 : hint_range(0.1, 5.0) = 3.5;
uniform float wave_speed_3 : hint_range(0.0, 3.0) = 1.2;
uniform vec2 wave_direction_3 = vec2(-0.6, 0.8);

// Normal map scrolling
uniform sampler2D normal_map_1 : hint_normal, filter_linear_mipmap, repeat_enable;
uniform sampler2D normal_map_2 : hint_normal, filter_linear_mipmap, repeat_enable;
uniform float normal_scale_1 : hint_range(0.5, 10.0) = 4.0;
uniform float normal_scale_2 : hint_range(0.5, 20.0) = 8.0;
uniform float normal_scroll_speed_1 : hint_range(0.0, 0.5) = 0.04;
uniform float normal_scroll_speed_2 : hint_range(0.0, 0.5) = 0.025;
uniform float normal_strength : hint_range(0.0, 2.0) = 0.8;

// Foam
uniform float foam_threshold : hint_range(0.0, 2.0) = 0.3;
uniform vec3 foam_color : source_color = vec3(0.85, 0.88, 0.82);
uniform float foam_intensity : hint_range(0.0, 1.0) = 0.4;

// Surface properties
uniform float roughness_base : hint_range(0.0, 1.0) = 0.05;
uniform float metallic : hint_range(0.0, 1.0) = 0.0;
uniform float specular : hint_range(0.0, 1.0) = 0.8;

// Edge/shore detection via depth texture
uniform sampler2D DEPTH_TEXTURE : hint_depth_texture, filter_linear_mipmap;

varying vec3 world_pos;

// Gerstner wave displacement
vec3 gerstner_wave(vec2 pos, float amplitude, float frequency, float speed, vec2 direction, float time) {
	vec2 dir = normalize(direction);
	float phase = dot(dir, pos) * frequency + time * speed;
	float steepness = 0.5; // Q factor
	float s = sin(phase);
	float c = cos(phase);
	return vec3(
		dir.x * amplitude * steepness * c,
		amplitude * s,
		dir.y * amplitude * steepness * c
	);
}

void vertex() {
	world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec2 xz = world_pos.xz;
	float t = TIME;

	// Sum Gerstner waves
	vec3 wave = vec3(0.0);
	wave += gerstner_wave(xz, wave_amplitude_1, wave_frequency_1, wave_speed_1, wave_direction_1, t);
	wave += gerstner_wave(xz, wave_amplitude_2, wave_frequency_2, wave_speed_2, wave_direction_2, t);
	wave += gerstner_wave(xz, wave_amplitude_3, wave_frequency_3, wave_speed_3, wave_direction_3, t);

	VERTEX.y += wave.y;
	VERTEX.x += wave.x;
	VERTEX.z += wave.z;

	// Recompute normal from wave derivatives
	float eps = 0.1;
	vec3 wave_px = gerstner_wave(xz + vec2(eps, 0.0), wave_amplitude_1, wave_frequency_1, wave_speed_1, wave_direction_1, t)
		+ gerstner_wave(xz + vec2(eps, 0.0), wave_amplitude_2, wave_frequency_2, wave_speed_2, wave_direction_2, t)
		+ gerstner_wave(xz + vec2(eps, 0.0), wave_amplitude_3, wave_frequency_3, wave_speed_3, wave_direction_3, t);
	vec3 wave_pz = gerstner_wave(xz + vec2(0.0, eps), wave_amplitude_1, wave_frequency_1, wave_speed_1, wave_direction_1, t)
		+ gerstner_wave(xz + vec2(0.0, eps), wave_amplitude_2, wave_frequency_2, wave_speed_2, wave_direction_2, t)
		+ gerstner_wave(xz + vec2(0.0, eps), wave_amplitude_3, wave_frequency_3, wave_speed_3, wave_direction_3, t);

	vec3 tangent_x = vec3(eps, wave_px.y - wave.y, 0.0);
	vec3 tangent_z = vec3(0.0, wave_pz.y - wave.y, eps);
	NORMAL = normalize(cross(tangent_z, tangent_x));
}

void fragment() {
	vec2 uv = world_pos.xz;

	// Scrolling dual normal maps for surface ripples
	vec2 scroll1 = vec2(TIME * normal_scroll_speed_1, TIME * normal_scroll_speed_1 * 0.3);
	vec2 scroll2 = vec2(-TIME * normal_scroll_speed_2 * 0.7, TIME * normal_scroll_speed_2);
	vec3 n1 = texture(normal_map_1, uv / normal_scale_1 + scroll1).rgb * 2.0 - 1.0;
	vec3 n2 = texture(normal_map_2, uv / normal_scale_2 + scroll2).rgb * 2.0 - 1.0;
	vec3 combined_normal = normalize(vec3(
		(n1.xy + n2.xy) * normal_strength,
		1.0));

	NORMAL_MAP = combined_normal * 0.5 + 0.5;
	NORMAL_MAP_DEPTH = normal_strength;

	// Depth-based color absorption (Beer's Law approximation)
	// Read scene depth for edge foam and transparency
	float depth_tex = texture(DEPTH_TEXTURE, SCREEN_UV).r;
	vec4 world_from_depth = INV_PROJECTION_MATRIX * vec4(SCREEN_UV * 2.0 - 1.0, depth_tex, 1.0);
	world_from_depth.xyz /= world_from_depth.w;
	float scene_depth = -world_from_depth.z;
	float frag_depth = FRAGCOORD.z / FRAGCOORD.w;
	float water_depth = max(scene_depth - frag_depth, 0.0) * depth_fade_distance;

	// Color blending based on water depth
	vec3 water_color = mix(shallow_color, deep_color, clamp(water_depth / color_depth_factor, 0.0, 1.0));

	// Fresnel reflection (more reflective at grazing angles)
	float ndv = max(dot(NORMAL, VIEW), 0.0);
	float fresnel = fresnel_bias + (1.0 - fresnel_bias) * pow(1.0 - ndv, fresnel_power);

	// Edge foam where water meets terrain
	float foam = 0.0;
	if (foam_threshold > 0.0) {
		foam = 1.0 - smoothstep(0.0, foam_threshold, water_depth);
		foam *= foam_intensity;
		// Add noise to foam for natural look
		float foam_noise = fract(sin(dot(uv * 5.0 + TIME * 0.1, vec2(12.9898, 78.233))) * 43758.5453);
		foam *= smoothstep(0.3, 0.7, foam_noise);
	}

	// Blend water color with foam
	vec3 final_color = mix(water_color, foam_color, foam);

	// Increase reflectivity with fresnel
	final_color = mix(final_color, final_color * 1.3, fresnel * 0.3);

	ALBEDO = final_color;
	ROUGHNESS = mix(roughness_base, 0.4, foam);
	METALLIC = metallic;
	SPECULAR = specular;
	ALPHA = clamp(base_alpha + fresnel * 0.3 + foam * 0.3, 0.0, 0.95);
}
