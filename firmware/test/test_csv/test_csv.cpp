/**
 * Pathfinder CSV Format Tests
 *
 * Validates expected CSV column structure for different pair counts.
 * The firmware generates headers at runtime, so these tests verify
 * the expected patterns match.
 *
 * Runs on host via PlatformIO native environment (no hardware needed).
 */

#include <unity.h>
#include <string.h>
#include <stdio.h>

// Simulate the firmware's runtime CSV header generation
// This mirrors writeCSVHeader() in main.cpp
static void build_csv_header(char *buf, size_t bufsize, int n_pairs, int gps_quality) {
    int pos = 0;
    pos += snprintf(buf + pos, bufsize - pos, "timestamp,lat,lon");
    if (gps_quality) {
        pos += snprintf(buf + pos, bufsize - pos, ",fix_quality,hdop,altitude");
    }
    for (int i = 1; i <= n_pairs; i++) {
        pos += snprintf(buf + pos, bufsize - pos, ",g%d_top,g%d_bot,g%d_grad", i, i, i);
    }
}

static int count_commas(const char *s) {
    int count = 0;
    while (*s) {
        if (*s == ',') count++;
        s++;
    }
    return count;
}

// Column count = 3 (base) + 3*pairs [+ 3 (gps quality)]
// Commas = columns - 1

void test_csv_4pair_no_gps_quality(void) {
    char buf[256];
    build_csv_header(buf, sizeof(buf), 4, 0);
    // 3 base + 12 pair = 15 columns = 14 commas
    TEST_ASSERT_EQUAL_INT(14, count_commas(buf));
    TEST_ASSERT_NOT_NULL(strstr(buf, "timestamp,lat,lon"));
    TEST_ASSERT_NOT_NULL(strstr(buf, "g4_grad"));
    TEST_ASSERT_NULL(strstr(buf, "fix_quality"));
}

void test_csv_4pair_with_gps_quality(void) {
    char buf[256];
    build_csv_header(buf, sizeof(buf), 4, 1);
    // 3 base + 3 gps + 12 pair = 18 columns = 17 commas
    TEST_ASSERT_EQUAL_INT(17, count_commas(buf));
    TEST_ASSERT_NOT_NULL(strstr(buf, "fix_quality,hdop,altitude"));
    TEST_ASSERT_NOT_NULL(strstr(buf, "g4_grad"));
}

void test_csv_2pair_no_gps_quality(void) {
    char buf[256];
    build_csv_header(buf, sizeof(buf), 2, 0);
    // 3 base + 6 pair = 9 columns = 8 commas
    TEST_ASSERT_EQUAL_INT(8, count_commas(buf));
    TEST_ASSERT_NOT_NULL(strstr(buf, "g2_grad"));
    TEST_ASSERT_NULL(strstr(buf, "g3_top"));
}

void test_csv_1pair(void) {
    char buf[256];
    build_csv_header(buf, sizeof(buf), 1, 0);
    // 3 base + 3 pair = 6 columns = 5 commas
    TEST_ASSERT_EQUAL_INT(5, count_commas(buf));
    TEST_ASSERT_NOT_NULL(strstr(buf, "g1_grad"));
    TEST_ASSERT_NULL(strstr(buf, "g2_top"));
}

void test_csv_starts_with_timestamp(void) {
    char buf[256];
    build_csv_header(buf, sizeof(buf), 2, 0);
    TEST_ASSERT_TRUE(strncmp(buf, "timestamp,", 10) == 0);
}

void test_csv_no_trailing_comma(void) {
    for (int pairs = 1; pairs <= 4; pairs++) {
        for (int gps = 0; gps <= 1; gps++) {
            char buf[256];
            build_csv_header(buf, sizeof(buf), pairs, gps);
            size_t len = strlen(buf);
            TEST_ASSERT_TRUE(len > 0);
            TEST_ASSERT_NOT_EQUAL(',', buf[len - 1]);
        }
    }
}

void test_csv_2pair_with_gps_quality(void) {
    char buf[256];
    build_csv_header(buf, sizeof(buf), 2, 1);
    // 3 base + 3 gps + 6 pair = 12 columns = 11 commas
    TEST_ASSERT_EQUAL_INT(11, count_commas(buf));
    TEST_ASSERT_NOT_NULL(strstr(buf, "fix_quality"));
    TEST_ASSERT_NOT_NULL(strstr(buf, "g2_grad"));
    TEST_ASSERT_NULL(strstr(buf, "g3_top"));
}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    RUN_TEST(test_csv_4pair_no_gps_quality);
    RUN_TEST(test_csv_4pair_with_gps_quality);
    RUN_TEST(test_csv_2pair_no_gps_quality);
    RUN_TEST(test_csv_1pair);
    RUN_TEST(test_csv_starts_with_timestamp);
    RUN_TEST(test_csv_no_trailing_comma);
    RUN_TEST(test_csv_2pair_with_gps_quality);
    return UNITY_END();
}
