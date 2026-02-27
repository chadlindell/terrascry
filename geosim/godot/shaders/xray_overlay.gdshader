shader_type spatial;
render_mode unshaded, depth_test_disabled, cull_back, blend_mix;

uniform vec4 overlay_color : source_color = vec4(0.9, 0.6, 0.2, 0.45);
uniform vec4 emission_color : source_color = vec4(0.8, 0.4, 0.1, 1.0);
uniform float emission_strength = 0.5;
uniform float fresnel_power = 2.0;

void fragment() {
	float fresnel = pow(1.0 - abs(dot(NORMAL, VIEW)), fresnel_power);
	float alpha = overlay_color.a + fresnel * 0.3;
	alpha = clamp(alpha, 0.0, 0.85);

	ALBEDO = overlay_color.rgb;
	ALPHA = alpha;
	EMISSION = emission_color.rgb * (emission_strength + fresnel * 0.3);
}
