/**
 * Pathfinder ADC Saturation Handling Tests
 *
 * Tests the checkSaturation() logic: readings above ADC_SATURATION_THRESHOLD
 * (default 32000) trigger a saturation counter. Both positive and negative
 * extremes are tested.
 *
 * Runs on host via PlatformIO native environment (no hardware needed).
 */

#include <unity.h>
#include <stdint.h>

// Reproduce the threshold from config.h
static const int16_t ADC_SATURATION_THRESHOLD = 32000;

// Reproduce saturation counter (mirrors global adcSaturationCount in main.cpp)
static uint16_t adcSaturationCount;

// Extracted from main.cpp checkSaturation()
static void checkSaturation(int16_t value) {
    if (value > ADC_SATURATION_THRESHOLD || value < -ADC_SATURATION_THRESHOLD) {
        adcSaturationCount++;
    }
}

void setUp(void) {
    adcSaturationCount = 0;
}

void tearDown(void) {
    // nothing
}

// --- Tests: positive saturation ---

void test_adc_at_threshold_triggers_saturation(void) {
    // Value of 32001 is above threshold -> should trigger
    checkSaturation(32001);
    TEST_ASSERT_EQUAL_UINT16(1, adcSaturationCount);
}

void test_adc_exactly_at_threshold_does_not_trigger(void) {
    // Value exactly at threshold boundary (32000) should NOT trigger
    // because the check is strictly greater than
    checkSaturation(32000);
    TEST_ASSERT_EQUAL_UINT16(0, adcSaturationCount);
}

void test_adc_well_below_threshold_does_not_trigger(void) {
    checkSaturation(16000);
    TEST_ASSERT_EQUAL_UINT16(0, adcSaturationCount);
}

void test_adc_zero_does_not_trigger(void) {
    checkSaturation(0);
    TEST_ASSERT_EQUAL_UINT16(0, adcSaturationCount);
}

void test_adc_max_positive_triggers(void) {
    checkSaturation(32767);
    TEST_ASSERT_EQUAL_UINT16(1, adcSaturationCount);
}

// --- Tests: negative saturation ---

void test_adc_negative_beyond_threshold_triggers(void) {
    checkSaturation(-32001);
    TEST_ASSERT_EQUAL_UINT16(1, adcSaturationCount);
}

void test_adc_negative_at_threshold_does_not_trigger(void) {
    // -32000 is NOT less than -32000, so should not trigger
    checkSaturation(-32000);
    TEST_ASSERT_EQUAL_UINT16(0, adcSaturationCount);
}

void test_adc_negative_below_threshold_does_not_trigger(void) {
    checkSaturation(-16000);
    TEST_ASSERT_EQUAL_UINT16(0, adcSaturationCount);
}

void test_adc_min_negative_triggers(void) {
    // INT16_MIN = -32768
    checkSaturation(-32768);
    TEST_ASSERT_EQUAL_UINT16(1, adcSaturationCount);
}

// --- Tests: counter increments ---

void test_saturation_count_increments_correctly(void) {
    checkSaturation(32500);
    checkSaturation(32100);
    checkSaturation(32767);
    TEST_ASSERT_EQUAL_UINT16(3, adcSaturationCount);
}

void test_saturation_count_mixed_pos_neg(void) {
    checkSaturation(32500);   // +sat
    checkSaturation(-32500);  // -sat
    checkSaturation(16000);   // no sat
    checkSaturation(-16000);  // no sat
    checkSaturation(32001);   // +sat
    TEST_ASSERT_EQUAL_UINT16(3, adcSaturationCount);
}

void test_saturation_count_no_false_positives(void) {
    // Run many values that are all within range
    for (int16_t v = -32000; v <= 32000; v += 1000) {
        checkSaturation(v);
    }
    TEST_ASSERT_EQUAL_UINT16(0, adcSaturationCount);
}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    RUN_TEST(test_adc_at_threshold_triggers_saturation);
    RUN_TEST(test_adc_exactly_at_threshold_does_not_trigger);
    RUN_TEST(test_adc_well_below_threshold_does_not_trigger);
    RUN_TEST(test_adc_zero_does_not_trigger);
    RUN_TEST(test_adc_max_positive_triggers);
    RUN_TEST(test_adc_negative_beyond_threshold_triggers);
    RUN_TEST(test_adc_negative_at_threshold_does_not_trigger);
    RUN_TEST(test_adc_negative_below_threshold_does_not_trigger);
    RUN_TEST(test_adc_min_negative_triggers);
    RUN_TEST(test_saturation_count_increments_correctly);
    RUN_TEST(test_saturation_count_mixed_pos_neg);
    RUN_TEST(test_saturation_count_no_false_positives);
    return UNITY_END();
}
