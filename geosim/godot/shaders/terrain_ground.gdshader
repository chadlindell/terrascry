shader_type spatial;

// Terrain ground shader: blends 6 terrain materials (grass, dirt, rock, mud, gravel, sand)
// based on surface normal, noise, and scenario-driven splatmap weights.
// Features: PBR texture splatting with scenario tint, patchy grass clumps with bare dirt,
// rock on steep slopes via triplanar mapping, expanded color palette, domain warping,
// normal map blending, moisture mask, anti-tiling, curvature AO, height-blend transitions,
// parallax occlusion mapping, and distance-based detail LOD.

// Grass colors (expanded palette) — act as tint multipliers when textures are present
uniform vec3 grass_color_1 : source_color = vec3(0.28, 0.42, 0.18);
uniform vec3 grass_color_2 : source_color = vec3(0.32, 0.50, 0.22);
uniform vec3 grass_color_3 : source_color = vec3(0.38, 0.45, 0.15); // yellow-green
uniform vec3 grass_color_4 : source_color = vec3(0.22, 0.35, 0.14); // dark green

// Dirt colors
uniform vec3 dirt_color : source_color = vec3(0.45, 0.35, 0.25);
uniform vec3 dirt_color_alt : source_color = vec3(0.40, 0.30, 0.20);
uniform vec3 dirt_color_tan : source_color = vec3(0.52, 0.42, 0.30); // sandy tan

// Rock colors
uniform vec3 rock_color : source_color = vec3(0.38, 0.36, 0.33);
uniform vec3 rock_color_alt : source_color = vec3(0.30, 0.28, 0.25);

// Mud colors
uniform vec3 mud_color : source_color = vec3(0.30, 0.22, 0.15);
uniform vec3 mud_color_alt : source_color = vec3(0.25, 0.18, 0.12);

// Gravel colors
uniform vec3 gravel_color : source_color = vec3(0.48, 0.45, 0.40);
uniform vec3 gravel_color_alt : source_color = vec3(0.42, 0.40, 0.36);

// Sand colors
uniform vec3 sand_color : source_color = vec3(0.72, 0.65, 0.50);
uniform vec3 sand_color_alt : source_color = vec3(0.65, 0.58, 0.42);

// Slope thresholds
uniform float grass_slope_threshold : hint_range(0.0, 1.0) = 0.85;
uniform float rock_slope_threshold : hint_range(0.0, 1.0) = 0.7;
uniform float slope_blend : hint_range(0.01, 0.3) = 0.1;

// Noise
uniform float noise_scale : hint_range(0.1, 20.0) = 3.0;

// Grass patchiness
uniform float grass_patch_scale : hint_range(0.5, 10.0) = 2.5;
uniform float grass_coverage : hint_range(0.0, 1.0) = 0.6;

// Moisture parameters
uniform float moisture_height_bias : hint_range(-5.0, 5.0) = 0.0;
uniform float moisture_strength : hint_range(0.0, 1.0) = 0.4;

// Normal perturbation strength
uniform float normal_strength : hint_range(0.0, 2.0) = 0.6;

// LOD distances
uniform float detail_near : hint_range(5.0, 30.0) = 15.0;
uniform float detail_far : hint_range(15.0, 60.0) = 35.0;

// Water table
uniform float water_table_height : hint_range(-10.0, 10.0) = -99.0;
uniform float water_table_blend : hint_range(0.05, 1.0) = 0.3;

// Extra material blend weights (scenario-driven, 0=absent, 1=dominant)
uniform float mud_weight : hint_range(0.0, 1.0) = 0.0;
uniform float gravel_weight : hint_range(0.0, 1.0) = 0.0;
uniform float sand_weight : hint_range(0.0, 1.0) = 0.0;

// PBR texture maps — grass
uniform sampler2D grass_albedo : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D grass_normal : hint_normal, filter_linear_mipmap, repeat_enable;
uniform sampler2D grass_roughness_tex : filter_linear_mipmap, repeat_enable;

// PBR texture maps — dirt
uniform sampler2D dirt_albedo : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D dirt_normal : hint_normal, filter_linear_mipmap, repeat_enable;
uniform sampler2D dirt_roughness_tex : filter_linear_mipmap, repeat_enable;

// PBR texture maps — rock
uniform sampler2D rock_albedo : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D rock_normal : hint_normal, filter_linear_mipmap, repeat_enable;
uniform sampler2D rock_roughness_tex : filter_linear_mipmap, repeat_enable;

// PBR texture maps — mud
uniform sampler2D mud_albedo : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D mud_normal : hint_normal, filter_linear_mipmap, repeat_enable;
uniform sampler2D mud_roughness_tex : filter_linear_mipmap, repeat_enable;

// PBR texture maps — gravel
uniform sampler2D gravel_albedo : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D gravel_normal : hint_normal, filter_linear_mipmap, repeat_enable;
uniform sampler2D gravel_roughness_tex : filter_linear_mipmap, repeat_enable;

// PBR texture maps — sand
uniform sampler2D sand_albedo : source_color, filter_linear_mipmap, repeat_enable;
uniform sampler2D sand_normal : hint_normal, filter_linear_mipmap, repeat_enable;
uniform sampler2D sand_roughness_tex : filter_linear_mipmap, repeat_enable;

// AO maps — crevice darkening per terrain layer
uniform sampler2D grass_ao : filter_linear_mipmap, repeat_enable;
uniform sampler2D dirt_ao : filter_linear_mipmap, repeat_enable;
uniform sampler2D rock_ao : filter_linear_mipmap, repeat_enable;
uniform sampler2D mud_ao : filter_linear_mipmap, repeat_enable;
uniform sampler2D gravel_ao : filter_linear_mipmap, repeat_enable;
uniform sampler2D sand_ao : filter_linear_mipmap, repeat_enable;

// Height maps — for height-based texture blending
uniform sampler2D grass_height : filter_linear_mipmap, repeat_enable;
uniform sampler2D dirt_height : filter_linear_mipmap, repeat_enable;
uniform sampler2D rock_height : filter_linear_mipmap, repeat_enable;
uniform sampler2D mud_height : filter_linear_mipmap, repeat_enable;
uniform sampler2D gravel_height : filter_linear_mipmap, repeat_enable;
uniform sampler2D sand_height : filter_linear_mipmap, repeat_enable;

// Displacement maps — for parallax occlusion mapping
uniform sampler2D grass_displacement : filter_linear_mipmap, repeat_enable;
uniform sampler2D dirt_displacement : filter_linear_mipmap, repeat_enable;

uniform float height_blend_sharpness : hint_range(1.0, 20.0) = 8.0;

// Detail normal — tiling micro-surface at close range
uniform sampler2D detail_normal : hint_normal, filter_linear_mipmap, repeat_enable;
uniform float detail_normal_scale : hint_range(0.1, 2.0) = 0.5;  // meters per repeat (high frequency)
uniform float detail_normal_strength : hint_range(0.0, 1.0) = 0.3;

// Texture scale (meters per repeat)
uniform float texture_scale : hint_range(0.5, 10.0) = 2.0;
uniform float rock_texture_scale : hint_range(0.5, 10.0) = 3.0;

// Parallax occlusion mapping
uniform float parallax_depth : hint_range(0.0, 0.1) = 0.03;
uniform int parallax_steps : hint_range(4, 16) = 8;

// Reference colors for tint normalization (default albedo middle-gray for each type)
const vec3 GRASS_REF = vec3(0.30, 0.45, 0.18);
const vec3 DIRT_REF = vec3(0.45, 0.35, 0.25);
const vec3 ROCK_REF = vec3(0.35, 0.33, 0.30);
const vec3 MUD_REF = vec3(0.30, 0.22, 0.15);
const vec3 GRAVEL_REF = vec3(0.48, 0.45, 0.40);
const vec3 SAND_REF = vec3(0.72, 0.65, 0.50);

// Hash-based noise
float hash(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

float hash2(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float value_noise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);

	float a = hash(i);
	float b = hash(i + vec2(1.0, 0.0));
	float c = hash(i + vec2(0.0, 1.0));
	float d = hash(i + vec2(1.0, 1.0));

	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
	float val = 0.0;
	float amp = 0.5;
	for (int i = 0; i < 3; i++) {
		val += amp * value_noise(p);
		p *= 2.1;
		amp *= 0.5;
	}
	return val;
}

float fbm4(vec2 p) {
	float val = 0.0;
	float amp = 0.5;
	for (int i = 0; i < 4; i++) {
		val += amp * value_noise(p);
		p *= 2.1;
		amp *= 0.5;
	}
	return val;
}

// Voronoi-like cell noise for grass patches
float voronoi(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	float min_dist = 1.0;
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			vec2 neighbor = vec2(float(x), float(y));
			vec2 point = vec2(hash(i + neighbor), hash2(i + neighbor));
			vec2 diff = neighbor + point - f;
			min_dist = min(min_dist, dot(diff, diff));
		}
	}
	return sqrt(min_dist);
}

// Triplanar sampling helpers for rock textures (avoids UV stretching on slopes)
vec4 triplanar_sample(sampler2D tex, vec3 world_pos, vec3 world_normal, float scale) {
	vec3 blend = abs(world_normal);
	blend = pow(blend, vec3(4.0));
	blend /= (blend.x + blend.y + blend.z);

	vec4 x_proj = texture(tex, world_pos.yz / scale);
	vec4 y_proj = texture(tex, world_pos.xz / scale);
	vec4 z_proj = texture(tex, world_pos.xy / scale);

	return x_proj * blend.x + y_proj * blend.y + z_proj * blend.z;
}

vec3 triplanar_normal_sample(sampler2D tex, vec3 world_pos, vec3 world_normal, float scale) {
	vec3 blend = abs(world_normal);
	blend = pow(blend, vec3(4.0));
	blend /= (blend.x + blend.y + blend.z);

	vec3 x_proj = texture(tex, world_pos.yz / scale).xyz * 2.0 - 1.0;
	vec3 y_proj = texture(tex, world_pos.xz / scale).xyz * 2.0 - 1.0;
	vec3 z_proj = texture(tex, world_pos.xy / scale).xyz * 2.0 - 1.0;

	vec3 blended = x_proj * blend.x + y_proj * blend.y + z_proj * blend.z;
	return normalize(blended * 0.5 + 0.5);  // Back to 0-1 range for NORMAL_MAP
}

float triplanar_sample_r(sampler2D tex, vec3 world_pos, vec3 world_normal, float scale) {
	vec3 blend = abs(world_normal);
	blend = pow(blend, vec3(4.0));
	blend /= (blend.x + blend.y + blend.z);

	float x_proj = texture(tex, world_pos.yz / scale).r;
	float y_proj = texture(tex, world_pos.xz / scale).r;
	float z_proj = texture(tex, world_pos.xy / scale).r;

	return x_proj * blend.x + y_proj * blend.y + z_proj * blend.z;
}

// Height-blend: transitions follow physical height patterns instead of smooth gradients
float height_blend(float h1, float h2, float blend_factor, float sharpness) {
	float ma = max(h1 + blend_factor, h2 + (1.0 - blend_factor)) - 0.5 / sharpness;
	float b1 = max(h1 + blend_factor - ma, 0.0);
	float b2 = max(h2 + (1.0 - blend_factor) - ma, 0.0);
	return b1 / max(b1 + b2, 0.001);
}

// Parallax occlusion mapping — offset UVs based on view angle and displacement
vec2 parallax_offset(vec2 uv, vec3 view_dir_tangent, sampler2D disp_map, float depth, int steps) {
	if (depth < 0.001) return uv;
	float layer_depth = 1.0 / float(steps);
	float current_layer = 0.0;
	vec2 delta_uv = view_dir_tangent.xy / max(view_dir_tangent.z, 0.001) * depth / float(steps);
	vec2 current_uv = uv;
	float current_disp = texture(disp_map, current_uv).r;

	for (int i = 0; i < 16; i++) {
		if (i >= steps) break;
		if (current_layer >= current_disp) break;
		current_uv -= delta_uv;
		current_disp = texture(disp_map, current_uv).r;
		current_layer += layer_depth;
	}

	// Linear interpolation between last two layers
	vec2 prev_uv = current_uv + delta_uv;
	float after_depth = current_disp - current_layer;
	float before_depth = texture(disp_map, prev_uv).r - current_layer + layer_depth;
	float weight = after_depth / max(after_depth - before_depth, 0.001);
	return mix(current_uv, prev_uv, weight);
}

varying vec3 world_pos_3d;
varying vec3 world_normal_3d;

void vertex() {
	world_pos_3d = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	world_normal_3d = normalize((MODEL_MATRIX * vec4(NORMAL, 0.0)).xyz);
}

void fragment() {
	vec2 world_uv = world_pos_3d.xz;
	float world_y = world_pos_3d.y;

	// Distance for LOD
	float view_dist = length(world_pos_3d - CAMERA_POSITION_WORLD);
	float detail_factor = 1.0 - smoothstep(detail_near, detail_far, view_dist);

	// Parallax occlusion mapping (only at close range for performance)
	vec2 base_uv = world_uv / texture_scale;
	if (detail_factor > 0.3 && parallax_depth > 0.001) {
		// Approximate tangent-space view direction from world view
		vec3 view_ws = normalize(CAMERA_POSITION_WORLD - world_pos_3d);
		vec3 view_tangent = vec3(dot(view_ws, vec3(1.0, 0.0, 0.0)),
								 dot(view_ws, vec3(0.0, 0.0, 1.0)),
								 dot(view_ws, vec3(0.0, 1.0, 0.0)));
		float pom_strength = detail_factor * parallax_depth;
		base_uv = parallax_offset(base_uv, view_tangent, grass_displacement, pom_strength, parallax_steps);
	}

	// Domain warping for natural flow
	vec2 warp = vec2(
		fbm(world_uv * 0.5),
		fbm(world_uv * 0.5 + vec2(5.2, 1.3))
	);
	vec2 warped_uv = world_uv + warp * 0.5;

	// Noise layers
	float n = fbm(warped_uv * noise_scale);
	float n2 = fbm(warped_uv * noise_scale * 0.4 + vec2(17.3, 31.7));

	// Position-based hash for color variation
	float color_hash = hash(floor(world_uv * 0.3));

	// Anti-tiling: large-scale color modulation
	float n_large = fbm(world_uv * noise_scale * 0.15 + vec2(53.7, 91.2));

	// Anti-tiling UV offset per tile
	vec2 tile_id = floor(world_uv / texture_scale);
	vec2 anti_tile_offset = vec2(hash(tile_id), hash2(tile_id)) * 0.5;

	// === TEXTURE UVs ===
	vec2 grass_uv = base_uv + anti_tile_offset;
	vec2 dirt_uv = base_uv + anti_tile_offset * 0.7;
	vec2 mud_uv = base_uv + anti_tile_offset * 0.6;
	vec2 gravel_uv = base_uv + anti_tile_offset * 0.8;
	vec2 sand_uv = base_uv + anti_tile_offset * 0.9;

	// === GRASS TEXTURE SAMPLING ===
	vec3 grass_tex_albedo = texture(grass_albedo, grass_uv).rgb;
	float grass_tex_rough = texture(grass_roughness_tex, grass_uv).r;
	vec3 grass_tex_norm = texture(grass_normal, grass_uv).rgb;
	float grass_ao_val = texture(grass_ao, grass_uv).r;
	float grass_height_val = texture(grass_height, grass_uv).r;

	// === DIRT TEXTURE SAMPLING ===
	vec3 dirt_tex_albedo = texture(dirt_albedo, dirt_uv).rgb;
	float dirt_tex_rough = texture(dirt_roughness_tex, dirt_uv).r;
	vec3 dirt_tex_norm = texture(dirt_normal, dirt_uv).rgb;
	float dirt_ao_val = texture(dirt_ao, dirt_uv).r;
	float dirt_height_val = texture(dirt_height, dirt_uv).r;

	// === ROCK TEXTURE SAMPLING (triplanar) ===
	vec3 wn = world_normal_3d;
	vec3 rock_tex_albedo = triplanar_sample(rock_albedo, world_pos_3d, wn, rock_texture_scale).rgb;
	float rock_tex_rough = triplanar_sample_r(rock_roughness_tex, world_pos_3d, wn, rock_texture_scale);
	vec3 rock_tex_norm = triplanar_normal_sample(rock_normal, world_pos_3d, wn, rock_texture_scale);
	float rock_ao_val = triplanar_sample_r(rock_ao, world_pos_3d, wn, rock_texture_scale);
	float rock_height_val = triplanar_sample_r(rock_height, world_pos_3d, wn, rock_texture_scale);

	// === MUD TEXTURE SAMPLING ===
	vec3 mud_tex_albedo = texture(mud_albedo, mud_uv).rgb;
	float mud_tex_rough = texture(mud_roughness_tex, mud_uv).r;
	vec3 mud_tex_norm = texture(mud_normal, mud_uv).rgb;
	float mud_ao_val = texture(mud_ao, mud_uv).r;
	float mud_height_val = texture(mud_height, mud_uv).r;

	// === GRAVEL TEXTURE SAMPLING ===
	vec3 gravel_tex_albedo = texture(gravel_albedo, gravel_uv).rgb;
	float gravel_tex_rough = texture(gravel_roughness_tex, gravel_uv).r;
	vec3 gravel_tex_norm = texture(gravel_normal, gravel_uv).rgb;
	float gravel_ao_val = texture(gravel_ao, gravel_uv).r;
	float gravel_height_val = texture(gravel_height, gravel_uv).r;

	// === SAND TEXTURE SAMPLING ===
	vec3 sand_tex_albedo = texture(sand_albedo, sand_uv).rgb;
	float sand_tex_rough = texture(sand_roughness_tex, sand_uv).r;
	vec3 sand_tex_norm = texture(sand_normal, sand_uv).rgb;
	float sand_ao_val = texture(sand_ao, sand_uv).r;
	float sand_height_val = texture(sand_height, sand_uv).r;

	// === GRASS: Patchy clumps via Voronoi ===
	float vor = voronoi(world_uv * grass_patch_scale);
	float grass_mask = smoothstep(grass_coverage + 0.15, grass_coverage - 0.1, vor);

	// Select grass tint from expanded palette based on position
	vec3 grass_tint;
	if (color_hash < 0.25) {
		grass_tint = mix(grass_color_1, grass_color_2, n);
	} else if (color_hash < 0.5) {
		grass_tint = mix(grass_color_2, grass_color_3, n);
	} else if (color_hash < 0.75) {
		grass_tint = mix(grass_color_3, grass_color_4, n);
	} else {
		grass_tint = mix(grass_color_4, grass_color_1, n);
	}

	// Apply texture * tint (normalized by reference color), clamped to prevent white blowout
	vec3 grass = min(grass_tex_albedo * (grass_tint / GRASS_REF), vec3(1.0));

	// === DIRT: Between grass patches ===
	vec3 dirt_tint;
	if (color_hash > 0.5) {
		dirt_tint = mix(dirt_color, dirt_color_tan, n2);
	} else {
		dirt_tint = mix(dirt_color, dirt_color_alt, n2);
	}
	vec3 dirt = min(dirt_tex_albedo * (dirt_tint / DIRT_REF), vec3(1.0));

	// === MUD ===
	vec3 mud_tint = mix(mud_color, mud_color_alt, n);
	vec3 mud = min(mud_tex_albedo * (mud_tint / MUD_REF), vec3(1.0));

	// === GRAVEL ===
	vec3 gravel_tint = mix(gravel_color, gravel_color_alt, n2);
	vec3 gravel = min(gravel_tex_albedo * (gravel_tint / GRAVEL_REF), vec3(1.0));

	// === SAND ===
	vec3 sand_tint = mix(sand_color, sand_color_alt, n);
	vec3 sand = min(sand_tex_albedo * (sand_tint / SAND_REF), vec3(1.0));

	// Grass/dirt blend within flat areas — height-blend for natural transitions
	float hb_grass_dirt = height_blend(grass_height_val, dirt_height_val, grass_mask, height_blend_sharpness);
	vec3 flat_color = mix(dirt, grass, hb_grass_dirt);
	vec3 flat_norm = mix(dirt_tex_norm, grass_tex_norm, hb_grass_dirt);
	float flat_rough = mix(dirt_tex_rough, grass_tex_rough, hb_grass_dirt);
	float flat_ao = mix(dirt_ao_val, grass_ao_val, hb_grass_dirt);
	float flat_height = mix(dirt_height_val, grass_height_val, hb_grass_dirt);

	// === BLEND IN EXTRA MATERIALS (mud, gravel, sand) based on scenario weights ===
	// These are mixed in proportionally — noise-modulated for natural distribution
	float total_extra = mud_weight + gravel_weight + sand_weight;
	if (total_extra > 0.01) {
		// Noise-modulated zone masks for each extra material
		float mud_noise = fbm(world_uv * 1.8 + vec2(11.3, 7.7));
		float gravel_noise = fbm(world_uv * 2.3 + vec2(23.1, 19.5));
		float sand_noise = fbm(world_uv * 1.5 + vec2(37.9, 41.2));

		// Sharpen noise to create distinct patches
		float mud_mask = smoothstep(0.3, 0.7, mud_noise) * mud_weight;
		float gravel_mask = smoothstep(0.35, 0.65, gravel_noise) * gravel_weight;
		float sand_mask = smoothstep(0.4, 0.7, sand_noise) * sand_weight;

		// Normalize masks so they don't exceed 1.0 total
		float mask_total = mud_mask + gravel_mask + sand_mask;
		float extra_strength = min(mask_total, 0.85);  // Cap at 85% so some base shows
		if (mask_total > 0.01) {
			mud_mask = mud_mask / mask_total * extra_strength;
			gravel_mask = gravel_mask / mask_total * extra_strength;
			sand_mask = sand_mask / mask_total * extra_strength;
		}

		float base_remain = 1.0 - (mud_mask + gravel_mask + sand_mask);

		// Height-blend for mud into base
		if (mud_mask > 0.01) {
			float hb_mud = height_blend(flat_height, mud_height_val, 1.0 - mud_mask, height_blend_sharpness * 0.7);
			flat_color = mix(mud, flat_color, hb_mud);
			flat_norm = mix(mud_tex_norm, flat_norm, hb_mud);
			flat_rough = mix(mud_tex_rough, flat_rough, hb_mud);
			flat_ao = mix(mud_ao_val, flat_ao, hb_mud);
			flat_height = mix(mud_height_val, flat_height, hb_mud);
		}

		// Gravel blending
		if (gravel_mask > 0.01) {
			float hb_gravel = height_blend(flat_height, gravel_height_val, 1.0 - gravel_mask, height_blend_sharpness);
			flat_color = mix(gravel, flat_color, hb_gravel);
			flat_norm = mix(gravel_tex_norm, flat_norm, hb_gravel);
			flat_rough = mix(gravel_tex_rough, flat_rough, hb_gravel);
			flat_ao = mix(gravel_ao_val, flat_ao, hb_gravel);
			flat_height = mix(gravel_height_val, flat_height, hb_gravel);
		}

		// Sand blending
		if (sand_mask > 0.01) {
			float hb_sand = height_blend(flat_height, sand_height_val, 1.0 - sand_mask, height_blend_sharpness);
			flat_color = mix(sand, flat_color, hb_sand);
			flat_norm = mix(sand_tex_norm, flat_norm, hb_sand);
			flat_rough = mix(sand_tex_rough, flat_rough, hb_sand);
			flat_ao = mix(sand_ao_val, flat_ao, hb_sand);
		}
	}

	// === ROCK: For steep slopes ===
	vec3 rock_tint = mix(rock_color, rock_color_alt, n);
	vec3 rock = min(rock_tex_albedo * (rock_tint / ROCK_REF), vec3(1.0));

	// Apply anti-tiling modulation (clamped to prevent white patches)
	flat_color = min(flat_color * mix(0.9, 1.1, n_large), vec3(1.0));
	rock = min(rock * mix(0.92, 1.08, n_large), vec3(1.0));

	// Micro-detail grit (LOD-controlled)
	if (detail_factor > 0.1) {
		float grit = fbm4(world_uv * 40.0) * 0.5 + 0.5;
		flat_color = min(flat_color * mix(1.0, mix(0.9, 1.05, grit), detail_factor), vec3(1.0));
		rock = min(rock * mix(1.0, mix(0.92, 1.06, grit), detail_factor), vec3(1.0));
	}

	// === SLOPE-BASED BLENDING ===
	float slope_y = NORMAL.y;
	float grass_factor = smoothstep(grass_slope_threshold - slope_blend,
		grass_slope_threshold + slope_blend, slope_y);
	float rock_factor = 1.0 - smoothstep(rock_slope_threshold - slope_blend,
		rock_slope_threshold + slope_blend, slope_y);

	vec3 base_color = flat_color;
	vec3 base_norm = flat_norm;
	float base_rough = flat_rough;
	float base_ao = flat_ao;
	// Transition zone: flat_color -> dirt -> rock
	if (slope_y < grass_slope_threshold) {
		float hb_slope = height_blend(flat_height, dirt_height_val, grass_factor, height_blend_sharpness);
		base_color = mix(dirt, flat_color, hb_slope);
		base_norm = mix(dirt_tex_norm, flat_norm, hb_slope);
		base_rough = mix(dirt_tex_rough, flat_rough, hb_slope);
		base_ao = mix(dirt_ao_val, flat_ao, hb_slope);
	}
	// Rock takes over on steep slopes — height-blend for natural rock/soil transition
	float hb_rock = height_blend(base_ao, rock_height_val, 1.0 - rock_factor, height_blend_sharpness);
	base_color = mix(base_color, rock, 1.0 - hb_rock);
	base_norm = mix(base_norm, rock_tex_norm, 1.0 - hb_rock);
	base_rough = mix(base_rough, rock_tex_rough, 1.0 - hb_rock);
	base_ao = mix(base_ao, rock_ao_val, 1.0 - hb_rock);

	// === MOISTURE/WETNESS ===
	float moisture = smoothstep(1.0, -0.5, world_y - moisture_height_bias);
	float concavity_moisture = 1.0 - fbm(world_uv * noise_scale * 1.5 + vec2(7.7, 3.3));
	moisture = clamp(moisture * 0.7 + concavity_moisture * 0.3, 0.0, 1.0) * moisture_strength;
	base_color *= mix(1.0, 0.7, moisture);

	// === CURVATURE AO (LOD-controlled) ===
	if (detail_factor > 0.2) {
		float eps = 0.3;
		float h_center = fbm4(world_uv * noise_scale * 0.8);
		float h_px = fbm4((world_uv + vec2(eps, 0.0)) * noise_scale * 0.8);
		float h_nx = fbm4((world_uv - vec2(eps, 0.0)) * noise_scale * 0.8);
		float h_py = fbm4((world_uv + vec2(0.0, eps)) * noise_scale * 0.8);
		float h_ny = fbm4((world_uv - vec2(0.0, eps)) * noise_scale * 0.8);
		float curvature = (h_px + h_nx + h_py + h_ny - 4.0 * h_center);
		float ao = clamp(1.0 - curvature * 8.0, 0.6, 1.0);
		base_color *= mix(1.0, ao, detail_factor);
	}

	// === WATER TABLE / WATERLOGGED GROUND ===
	if (water_table_height > -50.0) {
		float below_water = smoothstep(water_table_height + water_table_blend,
			water_table_height - water_table_blend * 0.5, world_y);
		if (below_water > 0.01) {
			base_color = mix(base_color, base_color * vec3(0.4, 0.35, 0.3), below_water * 0.7);
			moisture = max(moisture, below_water * 0.9);
		}
	}

	ALBEDO = min(base_color, vec3(1.0));

	// === AO from texture maps ===
	AO = base_ao;
	AO_LIGHT_AFFECT = 0.6;

	// === ROUGHNESS (texture-sourced) ===
	float roughness_noise = fbm(world_uv * noise_scale * 1.2 + vec2(23.1, 47.9));
	float grass_roughness = mix(base_rough, mix(0.75, 0.85, roughness_noise), 0.3);
	float dirt_roughness = mix(base_rough, mix(0.85, 0.95, roughness_noise), 0.3);
	float rock_roughness = mix(base_rough, mix(0.88, 0.98, roughness_noise), 0.3);
	float final_roughness = mix(dirt_roughness, grass_roughness, grass_factor);
	final_roughness = mix(final_roughness, rock_roughness, rock_factor);
	final_roughness = mix(final_roughness, mix(0.55, 0.72, roughness_noise), moisture);
	ROUGHNESS = final_roughness;

	METALLIC = 0.0;

	// === CLEARCOAT (wet areas — subtle sheen, not mirror-like) ===
	CLEARCOAT = moisture * 0.35;
	CLEARCOAT_ROUGHNESS = 0.45;
	SPECULAR = mix(0.4, 0.5, moisture);

	// === NORMAL MAP from textures (LOD-faded) with detail normal overlay ===
	vec3 combined_norm = base_norm;
	if (detail_factor > 0.1) {
		vec2 detail_uv = world_uv / detail_normal_scale;
		vec3 detail_n = texture(detail_normal, detail_uv).xyz * 2.0 - 1.0;
		vec3 base_n = base_norm * 2.0 - 1.0;
		combined_norm = normalize(vec3(
			base_n.xy + detail_n.xy * detail_normal_strength * detail_factor,
			base_n.z)) * 0.5 + 0.5;
	}
	NORMAL_MAP = combined_norm;
	NORMAL_MAP_DEPTH = detail_factor * 0.8;

	// === NORMAL PERTURBATION (procedural, reduced by 50% — texture normals carry micro-detail) ===
	if (detail_factor > 0.1) {
		float n_eps = 0.15;
		float fn_center = fbm4(world_uv * noise_scale * 2.0);
		float fn_dx = fbm4((world_uv + vec2(n_eps, 0.0)) * noise_scale * 2.0);
		float fn_dz = fbm4((world_uv + vec2(0.0, n_eps)) * noise_scale * 2.0);
		vec3 perturb = vec3(
			(fn_dx - fn_center) / n_eps,
			0.0,
			(fn_dz - fn_center) / n_eps
		) * normal_strength * 0.5;

		float dist_fade = detail_factor;
		NORMAL = normalize(NORMAL + perturb * dist_fade);
	}
}
