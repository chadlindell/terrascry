/**
 * Pathfinder Gradient Calculation Tests
 *
 * Tests the core gradient computation logic: gradient = bottom - top.
 * Runs on host via PlatformIO native environment (no hardware needed).
 */

#include <unity.h>
#include <stdint.h>

// Gradient calculation extracted from readGradiometers()
static int16_t compute_gradient(int16_t top, int16_t bot) {
    return bot - top;
}

void test_gradient_positive_anomaly(void) {
    // Bottom sensor reads higher than top -> positive gradient (near-surface target)
    TEST_ASSERT_EQUAL_INT16(500, compute_gradient(12000, 12500));
}

void test_gradient_negative_anomaly(void) {
    // Bottom reads lower than top -> negative gradient
    TEST_ASSERT_EQUAL_INT16(-500, compute_gradient(12500, 12000));
}

void test_gradient_zero_field(void) {
    TEST_ASSERT_EQUAL_INT16(0, compute_gradient(0, 0));
}

void test_gradient_equal_readings(void) {
    // Both sensors read the same -> zero gradient (no anomaly)
    TEST_ASSERT_EQUAL_INT16(0, compute_gradient(16384, 16384));
}

void test_gradient_max_positive(void) {
    TEST_ASSERT_EQUAL_INT16(32767, compute_gradient(0, 32767));
}

void test_gradient_max_negative(void) {
    TEST_ASSERT_EQUAL_INT16(-32767, compute_gradient(32767, 0));
}

void test_gradient_small_difference(void) {
    // Typical quiet field: small gradient
    TEST_ASSERT_EQUAL_INT16(3, compute_gradient(16000, 16003));
}

void test_saturation_threshold(void) {
    const int16_t THRESHOLD = 32000;
    TEST_ASSERT_FALSE(16000 > THRESHOLD || 16000 < -THRESHOLD);
    TEST_ASSERT_TRUE(32500 > THRESHOLD || 32500 < -THRESHOLD);
    TEST_ASSERT_TRUE(-32500 > THRESHOLD || -32500 < -THRESHOLD);
}

void test_gradient_array_loop(void) {
    // Simulate the array-based readGradiometers loop for N pairs
    const int n_pairs = 4;
    int16_t top[4]  = {12000, 12100, 12200, 12300};
    int16_t bot[4]  = {12500, 12400, 12300, 12200};
    int16_t grad[4];

    for (int i = 0; i < n_pairs; i++) {
        grad[i] = compute_gradient(top[i], bot[i]);
    }

    TEST_ASSERT_EQUAL_INT16(500, grad[0]);
    TEST_ASSERT_EQUAL_INT16(300, grad[1]);
    TEST_ASSERT_EQUAL_INT16(100, grad[2]);
    TEST_ASSERT_EQUAL_INT16(-100, grad[3]);
}

void test_gradient_single_pair(void) {
    // Simulate 1-pair configuration
    const int n_pairs = 1;
    int16_t top[1] = {15000};
    int16_t bot[1] = {15200};
    int16_t grad[1];

    for (int i = 0; i < n_pairs; i++) {
        grad[i] = compute_gradient(top[i], bot[i]);
    }

    TEST_ASSERT_EQUAL_INT16(200, grad[0]);
}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    RUN_TEST(test_gradient_positive_anomaly);
    RUN_TEST(test_gradient_negative_anomaly);
    RUN_TEST(test_gradient_zero_field);
    RUN_TEST(test_gradient_equal_readings);
    RUN_TEST(test_gradient_max_positive);
    RUN_TEST(test_gradient_max_negative);
    RUN_TEST(test_gradient_small_difference);
    RUN_TEST(test_saturation_threshold);
    RUN_TEST(test_gradient_array_loop);
    RUN_TEST(test_gradient_single_pair);
    return UNITY_END();
}
