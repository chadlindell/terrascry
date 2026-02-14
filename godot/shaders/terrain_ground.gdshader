shader_type spatial;

// Terrain ground shader: blends grass, dirt, and rock based on surface normal and noise.
// Features: patchy grass clumps with bare dirt, rock on steep slopes, expanded color
// palette, domain warping, noise-driven roughness, normal perturbation, moisture mask,
// anti-tiling, curvature AO, and distance-based detail LOD.

// Grass colors (expanded palette)
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

varying vec3 world_pos_3d;

void vertex() {
	world_pos_3d = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
	vec2 world_uv = world_pos_3d.xz;
	float world_y = world_pos_3d.y;

	// Distance for LOD
	float view_dist = length(world_pos_3d - CAMERA_POSITION_WORLD);
	float detail_factor = 1.0 - smoothstep(detail_near, detail_far, view_dist);

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

	// === GRASS: Patchy clumps via Voronoi ===
	float vor = voronoi(world_uv * grass_patch_scale);
	// Grass where Voronoi distance is small (clump centers), dirt between
	float grass_mask = smoothstep(grass_coverage + 0.15, grass_coverage - 0.1, vor);

	// Select grass color from expanded palette based on position
	vec3 grass;
	if (color_hash < 0.25) {
		grass = mix(grass_color_1, grass_color_2, n);
	} else if (color_hash < 0.5) {
		grass = mix(grass_color_2, grass_color_3, n);
	} else if (color_hash < 0.75) {
		grass = mix(grass_color_3, grass_color_4, n);
	} else {
		grass = mix(grass_color_4, grass_color_1, n);
	}

	// === DIRT: Between grass patches ===
	vec3 dirt;
	if (color_hash > 0.5) {
		dirt = mix(dirt_color, dirt_color_tan, n2);
	} else {
		dirt = mix(dirt_color, dirt_color_alt, n2);
	}

	// Grass/dirt blend within flat areas
	vec3 flat_color = mix(dirt, grass, grass_mask);

	// === ROCK: For steep slopes ===
	vec3 rock = mix(rock_color, rock_color_alt, n);

	// Apply anti-tiling modulation
	flat_color *= mix(0.9, 1.1, n_large);
	rock *= mix(0.92, 1.08, n_large);

	// Micro-detail grit (LOD-controlled)
	if (detail_factor > 0.1) {
		float grit = fbm4(world_uv * 40.0) * 0.5 + 0.5;
		flat_color *= mix(1.0, mix(0.9, 1.05, grit), detail_factor);
		rock *= mix(1.0, mix(0.92, 1.06, grit), detail_factor);
	}

	// === SLOPE-BASED BLENDING ===
	float slope_y = NORMAL.y;
	// Grass/dirt on flat areas (slope_y > grass_slope_threshold)
	float grass_factor = smoothstep(grass_slope_threshold - slope_blend,
		grass_slope_threshold + slope_blend, slope_y);
	// Rock on steep slopes (slope_y < rock_slope_threshold)
	float rock_factor = 1.0 - smoothstep(rock_slope_threshold - slope_blend,
		rock_slope_threshold + slope_blend, slope_y);

	vec3 base_color = flat_color;
	// Transition zone: flat_color -> dirt -> rock
	if (slope_y < grass_slope_threshold) {
		// Increasingly dirty as slope increases
		base_color = mix(dirt, flat_color, grass_factor);
	}
	// Rock takes over on steep slopes
	base_color = mix(base_color, rock, rock_factor);

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

	ALBEDO = base_color;

	// === ROUGHNESS ===
	float roughness_noise = fbm(world_uv * noise_scale * 1.2 + vec2(23.1, 47.9));
	float grass_roughness = mix(0.75, 0.85, roughness_noise);
	float dirt_roughness = mix(0.85, 0.95, roughness_noise);
	float rock_roughness = mix(0.88, 0.98, roughness_noise);
	float base_roughness = mix(dirt_roughness, grass_roughness, grass_factor);
	base_roughness = mix(base_roughness, rock_roughness, rock_factor);
	base_roughness = mix(base_roughness, mix(0.3, 0.5, roughness_noise), moisture);
	ROUGHNESS = base_roughness;

	METALLIC = 0.0;

	// === CLEARCOAT (wet areas) ===
	CLEARCOAT = moisture * 0.7;
	CLEARCOAT_ROUGHNESS = 0.25;
	SPECULAR = mix(0.5, 0.6, moisture);

	// === NORMAL PERTURBATION (LOD-controlled) ===
	if (detail_factor > 0.1) {
		float n_eps = 0.15;
		float fn_center = fbm4(world_uv * noise_scale * 2.0);
		float fn_dx = fbm4((world_uv + vec2(n_eps, 0.0)) * noise_scale * 2.0);
		float fn_dz = fbm4((world_uv + vec2(0.0, n_eps)) * noise_scale * 2.0);
		vec3 perturb = vec3(
			(fn_dx - fn_center) / n_eps,
			0.0,
			(fn_dz - fn_center) / n_eps
		) * normal_strength;

		float dist_fade = detail_factor;
		NORMAL = normalize(NORMAL + perturb * dist_fade);
	}
}
