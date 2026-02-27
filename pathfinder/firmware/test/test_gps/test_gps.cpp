/**
 * Pathfinder GPS Data Validation Tests
 *
 * Tests GPS lock detection logic and coordinate validation.
 * The firmware determines gpsLocked from:
 *   gpsLocked = gps.location.isValid() && gps.location.age() < 2000
 * and sets lat/lon to 0.0 when unlocked.
 *
 * Runs on host via PlatformIO native environment (no hardware needed).
 */

#include <unity.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// Simulate the GPS lock logic from main.cpp loop()
static const uint32_t GPS_AGE_LIMIT_MS = 2000;

static bool gps_is_locked(bool location_valid, uint32_t location_age_ms) {
    return location_valid && (location_age_ms < GPS_AGE_LIMIT_MS);
}

// Simulate the coordinate assignment from main.cpp loop()
static void apply_gps_coords(bool locked, double gps_lat, double gps_lon,
                              double *out_lat, double *out_lon) {
    if (locked) {
        *out_lat = gps_lat;
        *out_lon = gps_lon;
    } else {
        *out_lat = 0.0;
        *out_lon = 0.0;
    }
}

// Validate coordinate ranges (utility for data quality checking)
static bool is_valid_latitude(double lat) {
    return (lat >= -90.0) && (lat <= 90.0);
}

static bool is_valid_longitude(double lon) {
    return (lon >= -180.0) && (lon <= 180.0);
}

void setUp(void) {
    // nothing
}

void tearDown(void) {
    // nothing
}

// --- Tests: GPS lock detection ---

void test_gps_locked_valid_and_fresh(void) {
    // Valid location with age 500ms -> locked
    TEST_ASSERT_TRUE(gps_is_locked(true, 500));
}

void test_gps_locked_age_zero(void) {
    // Valid location with age 0ms -> locked
    TEST_ASSERT_TRUE(gps_is_locked(true, 0));
}

void test_gps_locked_age_just_under_limit(void) {
    // Valid location with age 1999ms -> still locked
    TEST_ASSERT_TRUE(gps_is_locked(true, 1999));
}

void test_gps_unlocked_age_at_limit(void) {
    // Valid location but age exactly 2000ms -> unlocked (not < 2000)
    TEST_ASSERT_FALSE(gps_is_locked(true, 2000));
}

void test_gps_unlocked_stale_fix(void) {
    // Valid location but stale data (age 5000ms) -> unlocked
    TEST_ASSERT_FALSE(gps_is_locked(true, 5000));
}

void test_gps_unlocked_invalid_location(void) {
    // Invalid location even with fresh age -> unlocked
    TEST_ASSERT_FALSE(gps_is_locked(false, 100));
}

void test_gps_unlocked_invalid_and_stale(void) {
    // Invalid location AND stale -> definitely unlocked
    TEST_ASSERT_FALSE(gps_is_locked(false, 10000));
}

// --- Tests: coordinate range validation ---

void test_latitude_valid_positive(void) {
    TEST_ASSERT_TRUE(is_valid_latitude(51.5074));  // London
}

void test_latitude_valid_negative(void) {
    TEST_ASSERT_TRUE(is_valid_latitude(-33.8688)); // Sydney
}

void test_latitude_valid_zero(void) {
    TEST_ASSERT_TRUE(is_valid_latitude(0.0));      // Equator
}

void test_latitude_valid_north_pole(void) {
    TEST_ASSERT_TRUE(is_valid_latitude(90.0));
}

void test_latitude_valid_south_pole(void) {
    TEST_ASSERT_TRUE(is_valid_latitude(-90.0));
}

void test_latitude_invalid_too_high(void) {
    TEST_ASSERT_FALSE(is_valid_latitude(90.001));
}

void test_latitude_invalid_too_low(void) {
    TEST_ASSERT_FALSE(is_valid_latitude(-90.001));
}

void test_longitude_valid_positive(void) {
    TEST_ASSERT_TRUE(is_valid_longitude(139.6917)); // Tokyo
}

void test_longitude_valid_negative(void) {
    TEST_ASSERT_TRUE(is_valid_longitude(-73.9857)); // New York
}

void test_longitude_valid_zero(void) {
    TEST_ASSERT_TRUE(is_valid_longitude(0.0));      // Prime meridian
}

void test_longitude_valid_antimeridian_pos(void) {
    TEST_ASSERT_TRUE(is_valid_longitude(180.0));
}

void test_longitude_valid_antimeridian_neg(void) {
    TEST_ASSERT_TRUE(is_valid_longitude(-180.0));
}

void test_longitude_invalid_too_high(void) {
    TEST_ASSERT_FALSE(is_valid_longitude(180.001));
}

void test_longitude_invalid_too_low(void) {
    TEST_ASSERT_FALSE(is_valid_longitude(-180.001));
}

// --- Tests: zero coordinates when GPS unlocked ---

void test_coords_zero_when_unlocked(void) {
    double lat, lon;
    apply_gps_coords(false, 51.5074, -0.1278, &lat, &lon);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, lat);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, lon);
}

void test_coords_passed_through_when_locked(void) {
    double lat, lon;
    apply_gps_coords(true, 51.5074, -0.1278, &lat, &lon);
    TEST_ASSERT_DOUBLE_WITHIN(0.0001, 51.5074, lat);
    TEST_ASSERT_DOUBLE_WITHIN(0.0001, -0.1278, lon);
}

void test_coords_zero_when_invalid_and_stale(void) {
    bool locked = gps_is_locked(false, 5000);
    double lat, lon;
    apply_gps_coords(locked, 40.7128, -74.0060, &lat, &lon);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, lat);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, lon);
}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    RUN_TEST(test_gps_locked_valid_and_fresh);
    RUN_TEST(test_gps_locked_age_zero);
    RUN_TEST(test_gps_locked_age_just_under_limit);
    RUN_TEST(test_gps_unlocked_age_at_limit);
    RUN_TEST(test_gps_unlocked_stale_fix);
    RUN_TEST(test_gps_unlocked_invalid_location);
    RUN_TEST(test_gps_unlocked_invalid_and_stale);
    RUN_TEST(test_latitude_valid_positive);
    RUN_TEST(test_latitude_valid_negative);
    RUN_TEST(test_latitude_valid_zero);
    RUN_TEST(test_latitude_valid_north_pole);
    RUN_TEST(test_latitude_valid_south_pole);
    RUN_TEST(test_latitude_invalid_too_high);
    RUN_TEST(test_latitude_invalid_too_low);
    RUN_TEST(test_longitude_valid_positive);
    RUN_TEST(test_longitude_valid_negative);
    RUN_TEST(test_longitude_valid_zero);
    RUN_TEST(test_longitude_valid_antimeridian_pos);
    RUN_TEST(test_longitude_valid_antimeridian_neg);
    RUN_TEST(test_longitude_invalid_too_high);
    RUN_TEST(test_longitude_invalid_too_low);
    RUN_TEST(test_coords_zero_when_unlocked);
    RUN_TEST(test_coords_passed_through_when_locked);
    RUN_TEST(test_coords_zero_when_invalid_and_stale);
    return UNITY_END();
}
