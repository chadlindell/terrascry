// Compute shader: Magnetic dipole field calculation
//
// Calculates B = μ₀/4π [3(m·r̂)r̂ - m] / r³ for N dipole sources
// at M observation points. Runs on GPU for real-time visualization.
//
// This is the "fast approximate" path described in the architecture:
// same analytical dipole math as the Python engine, but on GPU
// for thousands of evaluation points per frame.
//
// Phase 3 implementation — placeholder structure for now.

#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Observation points (x, y, z, padding)
layout(set = 0, binding = 0, std430) restrict readonly buffer ObsPoints {
    vec4 obs_points[];
};

// Dipole sources: position (xyz) + padding, then moment (xyz) + padding
layout(set = 0, binding = 1, std430) restrict readonly buffer Sources {
    vec4 source_data[];  // Interleaved: [pos0, mom0, pos1, mom1, ...]
};

// Output: B field at each observation point (Bx, By, Bz, |B|)
layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {
    vec4 B_output[];
};

// Uniforms
layout(push_constant) uniform PushConstants {
    uint n_obs;       // Number of observation points
    uint n_sources;   // Number of dipole sources
} params;

// μ₀/(4π) = 1e-7 T·m/A
const float MU0_4PI = 1.0e-7;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.n_obs) return;

    vec3 r_obs = obs_points[idx].xyz;
    vec3 B_total = vec3(0.0);

    for (uint i = 0; i < params.n_sources; i++) {
        vec3 r_src = source_data[i * 2].xyz;
        vec3 moment = source_data[i * 2 + 1].xyz;

        vec3 r = r_obs - r_src;
        float r_mag = length(r);

        // Skip if too close (avoids singularity)
        if (r_mag < 0.01) continue;

        float r3 = r_mag * r_mag * r_mag;
        float r5 = r3 * r_mag * r_mag;

        // B = μ₀/4π [3(m·r)r/r⁵ - m/r³]
        float m_dot_r = dot(moment, r);
        B_total += MU0_4PI * (3.0 * m_dot_r * r / r5 - moment / r3);
    }

    B_output[idx] = vec4(B_total, length(B_total));
}
