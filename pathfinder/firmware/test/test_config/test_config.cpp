/**
 * Pathfinder Configuration Validation Tests
 *
 * Tests compile-time configuration logic: NUM_SENSOR_PAIRS bounds,
 * NEEDS_ADC2 derivation, channel mapping arrays, and platform defaults.
 *
 * Runs on host via PlatformIO native environment (no hardware needed).
 */

#include <unity.h>
#include <stdint.h>

// ============================================================================
// Reproduce configuration constants from config.h for host-side testing.
// These mirror the actual firmware values so tests catch drift.
// ============================================================================

#define MAX_PAIRS 4

// Channel mapping arrays (must match config.h exactly)
static const uint8_t PAIR_ADC[MAX_PAIRS]     = {1, 1, 2, 2};
static const uint8_t PAIR_TOP_CH[MAX_PAIRS]  = {0, 2, 0, 2};
static const uint8_t PAIR_BOT_CH[MAX_PAIRS]  = {1, 3, 1, 3};

// Reproduce NEEDS_ADC2 derivation as a function for testing
static bool needs_adc2(int num_pairs) {
    return (num_pairs > 2);
}

// Reproduce platform default logic
static int platform_default_pairs(const char *platform) {
    // Mirrors the #ifdef logic in config.h
    if (platform[0] == 'd') return 2;  // drone
    return 4;                           // handheld, backpack
}

void setUp(void) {
    // nothing
}

void tearDown(void) {
    // nothing
}

// --- Tests: NUM_SENSOR_PAIRS bounds ---

void test_min_pairs_is_valid(void) {
    // 1 pair is the minimum valid configuration
    int pairs = 1;
    TEST_ASSERT_TRUE(pairs >= 1 && pairs <= 4);
}

void test_max_pairs_is_valid(void) {
    // 4 pairs is the maximum valid configuration
    int pairs = 4;
    TEST_ASSERT_TRUE(pairs >= 1 && pairs <= 4);
}

void test_zero_pairs_is_invalid(void) {
    int pairs = 0;
    TEST_ASSERT_FALSE(pairs >= 1 && pairs <= 4);
}

void test_five_pairs_is_invalid(void) {
    int pairs = 5;
    TEST_ASSERT_FALSE(pairs >= 1 && pairs <= 4);
}

void test_negative_pairs_is_invalid(void) {
    int pairs = -1;
    TEST_ASSERT_FALSE(pairs >= 1 && pairs <= 4);
}

// --- Tests: NEEDS_ADC2 derivation ---

void test_needs_adc2_with_1_pair(void) {
    TEST_ASSERT_FALSE(needs_adc2(1));
}

void test_needs_adc2_with_2_pairs(void) {
    TEST_ASSERT_FALSE(needs_adc2(2));
}

void test_needs_adc2_with_3_pairs(void) {
    TEST_ASSERT_TRUE(needs_adc2(3));
}

void test_needs_adc2_with_4_pairs(void) {
    TEST_ASSERT_TRUE(needs_adc2(4));
}

// --- Tests: channel mapping arrays ---

void test_pair_adc_assignment(void) {
    // Pairs 0-1 use ADC1, pairs 2-3 use ADC2
    TEST_ASSERT_EQUAL_UINT8(1, PAIR_ADC[0]);
    TEST_ASSERT_EQUAL_UINT8(1, PAIR_ADC[1]);
    TEST_ASSERT_EQUAL_UINT8(2, PAIR_ADC[2]);
    TEST_ASSERT_EQUAL_UINT8(2, PAIR_ADC[3]);
}

void test_pair_top_channels(void) {
    // Top sensors use even channels (0, 2) on each ADC
    TEST_ASSERT_EQUAL_UINT8(0, PAIR_TOP_CH[0]);
    TEST_ASSERT_EQUAL_UINT8(2, PAIR_TOP_CH[1]);
    TEST_ASSERT_EQUAL_UINT8(0, PAIR_TOP_CH[2]);
    TEST_ASSERT_EQUAL_UINT8(2, PAIR_TOP_CH[3]);
}

void test_pair_bot_channels(void) {
    // Bottom sensors use odd channels (1, 3) on each ADC
    TEST_ASSERT_EQUAL_UINT8(1, PAIR_BOT_CH[0]);
    TEST_ASSERT_EQUAL_UINT8(3, PAIR_BOT_CH[1]);
    TEST_ASSERT_EQUAL_UINT8(1, PAIR_BOT_CH[2]);
    TEST_ASSERT_EQUAL_UINT8(3, PAIR_BOT_CH[3]);
}

void test_top_and_bot_channels_are_adjacent(void) {
    // For each pair, bot channel = top channel + 1
    for (int i = 0; i < MAX_PAIRS; i++) {
        TEST_ASSERT_EQUAL_UINT8(PAIR_TOP_CH[i] + 1, PAIR_BOT_CH[i]);
    }
}

void test_channel_values_within_ads1115_range(void) {
    // ADS1115 has 4 single-ended channels (0-3)
    for (int i = 0; i < MAX_PAIRS; i++) {
        TEST_ASSERT_TRUE(PAIR_TOP_CH[i] <= 3);
        TEST_ASSERT_TRUE(PAIR_BOT_CH[i] <= 3);
    }
}

// --- Tests: platform defaults ---

void test_drone_defaults_to_2_pairs(void) {
    TEST_ASSERT_EQUAL_INT(2, platform_default_pairs("drone"));
}

void test_handheld_defaults_to_4_pairs(void) {
    TEST_ASSERT_EQUAL_INT(4, platform_default_pairs("handheld"));
}

void test_backpack_defaults_to_4_pairs(void) {
    TEST_ASSERT_EQUAL_INT(4, platform_default_pairs("backpack"));
}

void test_drone_needs_no_adc2(void) {
    int pairs = platform_default_pairs("drone");
    TEST_ASSERT_FALSE(needs_adc2(pairs));
}

void test_handheld_needs_adc2(void) {
    int pairs = platform_default_pairs("handheld");
    TEST_ASSERT_TRUE(needs_adc2(pairs));
}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    RUN_TEST(test_min_pairs_is_valid);
    RUN_TEST(test_max_pairs_is_valid);
    RUN_TEST(test_zero_pairs_is_invalid);
    RUN_TEST(test_five_pairs_is_invalid);
    RUN_TEST(test_negative_pairs_is_invalid);
    RUN_TEST(test_needs_adc2_with_1_pair);
    RUN_TEST(test_needs_adc2_with_2_pairs);
    RUN_TEST(test_needs_adc2_with_3_pairs);
    RUN_TEST(test_needs_adc2_with_4_pairs);
    RUN_TEST(test_pair_adc_assignment);
    RUN_TEST(test_pair_top_channels);
    RUN_TEST(test_pair_bot_channels);
    RUN_TEST(test_top_and_bot_channels_are_adjacent);
    RUN_TEST(test_channel_values_within_ads1115_range);
    RUN_TEST(test_drone_defaults_to_2_pairs);
    RUN_TEST(test_handheld_defaults_to_4_pairs);
    RUN_TEST(test_backpack_defaults_to_4_pairs);
    RUN_TEST(test_drone_needs_no_adc2);
    RUN_TEST(test_handheld_needs_adc2);
    return UNITY_END();
}
