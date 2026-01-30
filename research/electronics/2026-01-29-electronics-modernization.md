# Research Report

# Technical Component Review: Modernization of HIRT Electronics

**Key Points**
*   **Instrumentation Amplifier Modernization:** The legacy AD620 (9 nV/√Hz) can be upgraded to the **Analog Devices AD8421** for a significantly lower noise floor (3 nV/√Hz), critical for Magnetic Induction Tomography (MIT) preamplification. For applications prioritizing DC stability over AC noise floor, the **Texas Instruments INA188** offers zero-drift topology, though with slightly higher broadband noise.
*   **Operational Amplifier Efficiency:** The **Texas Instruments OPA2186** is a superior alternative to the OPA2277, offering zero-drift precision with a 90% reduction in quiescent current (90 µA vs. 800 µA), ideal for battery-operated hubs.
*   **Signal Generation & Acquisition:** The **Analog Devices AD9837** provides a footprint-optimized, lower-power alternative to the AD9833. For analog-to-digital conversion, the **Analog Devices AD7124-8** offers vastly improved integration (internal AFE) and power management compared to the ADS1256, while the **TI ADS1261** serves as a high-performance direct successor.
*   **Interconnect Ruggedization:** The **Weipu SP29 Series (26-pin)** is identified as the optimal IP68 replacement for DB25 connectors, offering high pin density, cost-effectiveness, and field-proven ruggedness suitable for the HIRT environment.

---

## 1. Introduction

This report provides a technical review and modernization strategy for the electronic components outlined in the HIRT (High-Resolution Induction Tomography) hardware documentation. The analysis focuses on the `hardware/bom/base-hub-bom.md` and `docs/hirt-whitepaper/sections/06-electronics-circuits.qmd` files, identifying legacy active components and proposing modern alternatives (2024–2026 lifecycle) from major suppliers such as Texas Instruments (TI) and Analog Devices (ADI).

The primary design goals for this modernization are reducing the noise floor—a critical parameter for the MIT receiver chain—minimizing power consumption for field deployment, and improving system integration. Additionally, this review addresses the specific requirement for "Zero-Drift Instrumentation Amplifiers" and recommends ruggedized IP68 interconnect solutions to replace non-waterproof D-Sub interfaces.

## 2. Analysis of Key Active Components

The following components were identified as the core active elements in the existing HIRT Base Hub design. Their current specifications establish the baseline for performance improvements.

### 2.1 AD620: Instrumentation Amplifier (RX Preamp)
*   **Function:** Acts as the primary preamplifier for the MIT receiver coil, amplifying millivolt-level induced signals [cite: 1].
*   **Current Specs:**
    *   **Noise:** 9 nV/√Hz at 1 kHz [cite: 2, 3].
    *   **Power:** 1.3 mA maximum supply current [cite: 2].
    *   **Architecture:** Classic 3-op-amp topology [cite: 4].
*   **Limitation:** While accurate, the AD620 is a legacy component (introduced decades ago). Its noise floor, while respectable, is outperformed by modern process technologies, and it lacks the zero-drift characteristics desirable for eliminating 1/f noise in low-frequency measurements.

### 2.2 OPA2277: Operational Amplifier (TX Driver)
*   **Function:** Drives the MIT transmit coils and serves as a precision buffer [cite: 1].
*   **Current Specs:**
    *   **Precision:** Ultralow offset voltage (10 µV) and drift (0.1 µV/°C) [cite: 5, 6].
    *   **Power:** 800 µA per amplifier quiescent current [cite: 5].
    *   **Bandwidth:** 1 MHz [cite: 5].
*   **Limitation:** The quiescent current is relatively high by modern standards for battery-powered precision applications.

### 2.3 AD9833: DDS Waveform Generator
*   **Function:** Generates the sine wave excitation signal (2–50 kHz) for the MIT transmitter [cite: 1, 7].
*   **Current Specs:**
    *   **Power:** 12.65 mW at 3 V [cite: 7].
    *   **Package:** 10-lead MSOP [cite: 7].
    *   **Resolution:** 28-bit frequency resolution [cite: 7].
*   **Limitation:** While still active, newer lithography allows for smaller footprints and marginally lower power consumption in similar DDS families.

### 2.4 ADS1256: Analog-to-Digital Converter
*   **Function:** High-resolution 24-bit digitization of the demodulated signals [cite: 1, 8].
*   **Current Specs:**
    *   **Sample Rate:** Up to 30 kSPS [cite: 9].
    *   **Power:** ~36 mW typical [cite: 9].
    *   **Noise:** Extremely low noise, but requires external signal conditioning (buffer/PGA) for optimal performance [cite: 8].
*   **Limitation:** Requires significant external circuitry (reference buffers, input drivers) to achieve datasheet performance, increasing PCB footprint and power budget [cite: 10].

---

## 3. Modern Component Alternatives (2024–2026)

This section details specific replacements selected for lower noise, reduced power, and "Active" lifecycle status.

### 3.1 Receiver Preamp Upgrade (Replacing AD620)

The MIT receiver chain requires the lowest possible voltage noise to resolve weak secondary fields. Two distinct upgrade paths are presented: one prioritizing absolute lowest noise (AC performance) and one prioritizing DC stability (Zero-Drift), as requested.

#### Option A: Lowest Noise Performance (Recommended for MIT)
**Component:** **Analog Devices AD8421**
*   **Description:** The AD8421 is a high-speed, ultralow-noise instrumentation amplifier designed specifically for signal conditioning where noise performance is limited only by the sensor [cite: 11, 12].
*   **Comparison to AD620:**
    *   **Noise Floor:** **3 nV/√Hz** vs. 9 nV/√Hz (AD620). This represents a **3x improvement** in signal-to-noise ratio potential [cite: 11, 13].
    *   **Power:** 2.0 mA quiescent current (slightly higher than AD620's 1.3 mA, but justified by noise reduction) [cite: 12].
    *   **Bandwidth:** 10 MHz (G=1) vs. 1 MHz (AD620), allowing for better phase linearity at 50 kHz [cite: 12].
*   **Lifecycle:** Active.

#### Option B: Zero-Drift Topology (Specific Request)
**Component:** **Texas Instruments INA188**
*   **Description:** The INA188 uses auto-zeroing techniques to achieve near-zero offset drift and eliminates 1/f noise, which is critical if the measurement bandwidth extends down to DC or very low frequencies [cite: 14, 15].
*   **Comparison to AD620:**
    *   **Architecture:** **Zero-Drift** (Chopper stabilized) vs. Bipolar.
    *   **Drift:** 0.2 µV/°C max vs. 0.6 µV/°C (AD620) [cite: 2, 14].
    *   **Noise:** **12 nV/√Hz** broadband. Note that while this eliminates low-frequency 1/f noise, the broadband white noise is *higher* than the AD620 (9 nV/√Hz) and significantly higher than the AD8421 [cite: 14].
    *   **Power:** 1.4 mA quiescent current [cite: 14].
*   **Trade-off Note:** For MIT applications operating at 2–50 kHz, the 1/f noise suppression of the INA188 is less beneficial than the ultra-low broadband noise of the AD8421. The AD8421 is recommended unless DC accuracy is the primary measurement parameter.

### 3.2 Transmit Driver Upgrade (Replacing OPA2277)

**Component:** **Texas Instruments OPA2186**
*   **Description:** A modern, 24-V, rail-to-rail input/output, zero-drift operational amplifier designed for high-precision portable instrumentation [cite: 16, 17].
*   **Comparison to OPA2277:**
    *   **Power Consumption:** **90 µA** per amplifier vs. 800 µA (OPA2277). This is an **89% reduction** in power consumption, significantly extending battery life in the hub [cite: 5, 17].
    *   **Precision:** 10 µV max offset and 0.04 µV/°C drift (Zero-Drift topology) ensures superior stability over temperature compared to the OPA2277's 0.1 µV/°C [cite: 17].
    *   **Footprint:** Available in smaller packages (SOT-23, VSSOP) compared to the standard SOIC/DIP of the OPA2277 [cite: 17].

### 3.3 DDS Generator Upgrade (Replacing AD9833)

**Component:** **Analog Devices AD9837**
*   **Description:** A low-power, programmable waveform generator that is functionally equivalent to the AD9833 but optimized for size and power [cite: 18, 19].
*   **Comparison to AD9833:**
    *   **Power:** **8.5 mW** at 2.3 V vs. 12.65 mW (AD9833).
    *   **Footprint:** Available in a **10-lead LFCSP** (3mm x 3mm), which is significantly smaller than the AD9833's MSOP package, aiding in high-density PCB layout [cite: 19].
    *   **Compatibility:** Uses the same SPI interface and register structure, simplifying firmware migration [cite: 18].

### 3.4 ADC Upgrade (Replacing ADS1256)

**Component:** **Analog Devices AD7124-8**
*   **Description:** A completely integrated Analog Front End (AFE) containing a low-noise, 24-bit Sigma-Delta ADC. It integrates the signal chain components that are external on the ADS1256 [cite: 20].
*   **Comparison to ADS1256:**
    *   **Integration:** Includes an internal rail-to-rail analog buffer, programmable gain amplifier (PGA), and reference buffers. The ADS1256 requires external buffers to achieve high impedance, which adds noise and power [cite: 8, 20].
    *   **Power:** Features three selectable power modes. In **Low Power Mode**, it consumes ~**255 µA** (approx. 1 mW), compared to the ADS1256's ~36 mW. Even in full power mode, it is highly efficient [cite: 20, 21].
    *   **Noise:** 24 nV RMS noise at low output rates, comparable to the ADS1256 but achievable with far fewer external components [cite: 21].

**Alternative (Direct TI Successor):** **ADS1261**
*   If staying within the TI ecosystem is preferred, the ADS1261 offers lower noise (0.34 µV RMS) and integrated features similar to the AD7124, representing a generational leap over the ADS1256 [cite: 22].

---

## 4. Zero-Drift Instrumentation Amplifier Review

The user specifically requested "Zero-Drift Instrumentation Amplifiers" to replace the AD620/INA128. It is important to distinguish between zero-drift *operational amplifiers* (used to build in-amps) and monolithic zero-drift *instrumentation amplifiers*.

### 4.1 Technology Overview
Standard instrumentation amplifiers (like the AD620) use bipolar transistors and laser trimming to achieve low offset. Zero-drift amplifiers use chopper-stabilization or auto-zeroing architectures to continuously correct offset voltages, effectively eliminating 1/f (pink) noise and thermal drift [cite: 23, 24].

### 4.2 Recommended Zero-Drift Replacement: TI INA188
As identified in Section 3.1, the **INA188** is the premier candidate for this specific request.
*   **Zero-Drift Architecture:** Eliminates 1/f noise corners, making it ideal for DC and near-DC measurements [cite: 14].
*   **Drift Specs:** 0.2 µV/°C offset drift is vastly superior to the AD620's 0.6 µV/°C [cite: 2, 14].
*   **Application Suitability:** While excellent for temperature stability, the chopping action creates noise spikes at the chopping frequency. For the HIRT application (2–50 kHz), designers must ensure the chopping frequency is well outside the measurement band or adequately filtered. The AD8421 remains the superior choice for *AC noise performance*, while the INA188 is the choice for *DC stability*.

### 4.3 Alternative: Programmable Gain Zero-Drift (ADA4254)
*   **Component:** **Analog Devices ADA4254**
*   **Features:** This is a Zero-Drift, High Voltage, Programmable Gain Instrumentation Amplifier (PGIA). It offers integrated gain switching (1/16 V/V to 128 V/V) and zero-drift topology [cite: 25, 26].
*   **Benefit:** It integrates the multiplexer and gain stages often found in tomography hubs, potentially replacing both the preamp and the multiplexers mentioned in the whitepaper [cite: 25].

---

## 5. Interconnect Modernization: IP68 Circular Connectors

The current design uses DB25 connectors [cite: 1], which are not waterproof and are prone to corrosion in field environments. The replacement must be IP68 rated, cost-effective, and support at least 25 pins (matching the DB25 pin count).

### 5.1 Recommended Solution: Weipu SP29 Series (26-Pin)
The **Weipu SP29** series is identified as the most cost-effective and geometrically suitable replacement.

*   **Pin Count:** Available in a **26-pin** configuration, which directly accommodates the 25 lines from the DB25 with one spare pin [cite: 27, 28].
*   **Ruggedness:** **IP68 rated** with a threaded coupling mechanism that ensures a watertight seal even under vibration and submersion [cite: 29].
*   **Cost-Effectiveness:** Constructed from high-grade Nylon66 (plastic shell), it is significantly cheaper than metal-shell mil-spec connectors (e.g., Amphenol D38999) while providing sufficient durability for field hubs [cite: 27, 30].
*   **Specifications:**
    *   **Panel Cutout:** 29mm [cite: 27].
    *   **Termination:** Solder [cite: 27].
    *   **Current Rating:** 5A per contact (sufficient for signal and low-power drive lines) [cite: 29].

### 5.2 Alternative: CNLINKO LP-24 / DH-24
*   **Pin Count:** Maxes out at **24 pins** [cite: 31, 32].
*   **Limitation:** This would require reducing the conductor count by one compared to the DB25, which may not be feasible depending on the grounding/shielding scheme of the HIRT probes. However, the locking mechanism (snap-lock) is faster than the threaded Weipu SP29.
*   **Recommendation:** Stick to the **Weipu SP29 (26-pin)** to ensure full compatibility with the existing conductor count.

## 6. Summary of Recommendations

| Legacy Component | Recommended Replacement | Key Advantage | Manufacturer |
| :--- | :--- | :--- | :--- |
| **AD620** | **AD8421** | **3x Lower Noise** (3 nV/√Hz), higher bandwidth. | Analog Devices |
| *(Alt for Zero-Drift)* | **INA188** | **Zero-Drift**, eliminates 1/f noise, high DC precision. | Texas Instruments |
| **OPA2277** | **OPA2186** | **90% Power Reduction** (90 µA), Zero-Drift. | Texas Instruments |
| **AD9833** | **AD9837** | **Smaller Footprint** (LFCSP), lower power (8.5 mW). | Analog Devices |
| **ADS1256** | **AD7124-8** | **Integrated AFE**, 3 power modes, drastically lower power. | Analog Devices |
| **DB25 Connector** | **Weipu SP29 (26-Pin)** | **IP68 Waterproof**, rugged threaded coupling, cost-effective. | Weipu |

This modernization strategy shifts the HIRT Base Hub towards a lower-power, higher-precision architecture suitable for long-term field deployment in challenging environments.

**Sources:**
1. base-hub-bom.md (fileSearchStores/hirt-project-xauxhhw4t934)
2. [ppu.edu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF5tqMgtAZgrG_E38srXlqRpMRPMSE-kErRMfEUMNoLE5_dJGMY-x4C-O8NLLFV7nBRN5WUv2l5r2HNBPId2WH6Cte2A3PFQB3xfKeCjsZv3RSRJoGhX0c9JAjLqYe-pHecfC5VUK8e4oJ4hCTvJwuYlFm7Wx14J1DIDo_9BofWcFAR60Ye-q5qdPieRSJMOr0=)
3. [analog.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHDa7yZemy3NDKNZ6DKer3461tFdmpNJlNLwsSIxQNF2Q9oqZAxoMp-coCRXp4rNO5O3O8AndmRTBiimwe6odIJ-T_lySVKLAhMQI5Fl1EYm1Y4dfjwvod5Bs-EJVnjaU4AYDA=)
4. [allelcoelec.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8gxoN9OKoaMaqB-VjnT1stbxdH4Om8Yp8t-sgjhFXkHkYTOw0q69YOBF2OStSa-2cW-YiOeqo1uCDSydTQ7VyCcTdHO9TbQgD6ylCE47cXGmrC8K9QDKGfsPt13odBetlKqrwa2y4K1fLkEmAvX3uJMZIUWiBpi53U121DoxCkcM6HGITlpNZl3Ie1GYLzmK0FI7tM_723mLoAw5exfS2R2p5K439r6nlRgj3p5s=)
5. [ti.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEnYLJl1gB0J53AvVCBRh1d4eKkQ_DbBId_6nHyDRdzB4yTP5hmm-Du2FUAHz1xS72ScVANg6mNjVhu0UoNS_Essa1yLXV9SXZvi2RYSZp2FkSIZdaXOTL_KRDQ)
6. [cdn-reichelt.de](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGsU7MnlfzIC0IMnbDU80XtcUvOhSeYUU_hAht_GUIvbrVCCInYaDB5K3IvwXYe1wwdBW9jytx68Cc706ZupcANv_Kbbp7t8pTddS_rEDs5kocGor62-1QZzxYHVizb53EmytT764O-OI5EwSlnrLmv6s2BF9w7QQ==)
7. [analog.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEE75prjNlijqo32lg9OK_GkAvVZ0yDnVZy1pHUJQhPrf_PEGkSxazg-hFM5uxPj7z47bC0UVn4hX9T8p5l_4nr5Tb8LjclpeN5ovSrRtGz5dyXAOEClzIFMdNLauIZPZ4RLTqi0VaFB_1d-A0vC6ZneXq228BSoFI_CPIch5PUMVR0vXQ=)
8. [ti.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGUXsotge-QZBwmEla607wzf6U2nUNAijFmyFGX-A06erhCQnAzdEUoOSiUNB1D0Yd3FJiywcgHgd53gWpeWS0YiKbbIjKFHUuCU6JUZ4_kPecOg0OpMadEZ8p3An1I8u5s2jU=)
9. [digchip.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0mzJW1aZHZjKEy-SvYlPaMxg-o6zIxEgH-2zH2DMAQMC1t4ky3LLsFT-7bKM4FtUM2B2aY6_MCOcrWgK0OS8QE9w6dyxbDteajkBAQTA70FqPWO2ij815ZiF6GjFxHEssooMXndLmafgUV7Di6p-rIINq6pVzK1tGGw==)
10. [eevblog.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCzZVBPXEBKMQBs9jmKzoI275FlYjwFjXOuA7nNhEEYXNgBdI0-MAqpN-_03mC1e7dkT9alTV2QnZ04b_4GIZVwzPTm5nOBxJsBLQzmyo4P3bV7Qe3EQ5NFRtigS3UXKESeTxL-0GvpUi6AbtMJQtNr3HoPgiVbg==)
11. [icdrex.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQET30Vsn8TrRwixBeurkBeb-VodTutH7wJzdiv1WYJUMei7jUZyBKsQcgwjsDl1Xw-3lI7BYTg29KABy5mUrseemXDmVnzmEHd5B_s4kH2fxHjTUz3siEkWusG-DN-tDTi6pOiFBhFNyQnYoMWhdicOBIIsQ-UZ_bE92ewBQJxQxDPufcVQSF29ZM9VOEfOxF61ZhoC)
12. [radiolocman.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGnHiZVeUaI8FXCSodKtHehmHqIQPYl20X8lWDPZmiXCNY_KNvTfZajopFPXPBtjbyyu6354qJP_TZl7E2k2JFbhE0T_RBH-Sd8Pbj4tVfMW99AWjgdCz0gyc7LrrHs3UbVvDRjlFcsE9Wt3nipWQ==)
13. [analog.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHtP2eTA9ej5P_p6aIYM5s2VbZfrCUHrc_6Ce6MMSYLJ3Gtccc-WOm0nZeT_Pib9DeN7UB9OKRh6kUfqvl9PpAh_DDsPSlzwMwZ8_Dqg7qStS1_xrfG5aNdFScYBihFJ2fLYsoDcRZ43krfyDgYj9yB0t7o8hAUBhTtfcRtBrVbDWoPnuU=)
14. [ti.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHAb2n2jcYtFPeHOq6ybDPE0rU9EYcgay0QhgqCWNB3a6rB7JwurbvD1RMEg5_J6Qw5Ugzsgu9KYsjMsydjRyq-W7b1zDfI75NRyAzIe_qgOcYBdJjiQNU=)
15. [alldatasheet.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGqXF9wlmdypSCaIejPBdZTnf5XPgyjconnqvuPOik1dJ72ljw2oILaeOJBcTrb1G5WqZvNosJUCsbrN8YN3-BDyiplpAk8bDcFs65w5wqJBaCzzEiR9JmbsU56R_FlbZ2c1asB1Y9hypGW5UODm_IoojjxpsTeUZY-BvMR)
16. [mouser.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGvcY99N14-CWxpfe6fNxSiZJOBSeqB_5u43F5GmoqcWIdR2zSKaTc5cgt_WBJReIp9DKqJS8y8nTGsbGaea9WORmtvSvohbysK6yfhHlkBERvEuGklsHfdKUuNxnh8xyS6XYAIWgdQOQ3y0yeDThmBTVY=)
17. [ti.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHnZTheHWz11cb2PjVC4CZifY8vAjIHHAup_S6JwiecyO_vYHt23-L5QUlg_Ga4I9OwOpp5-uuIUDqImQk7pPurCdZiiYKsMZJYEK422A4HtecFZHHc0YLFlSVkJgQCfq9LqmM=)
18. [eldruin.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE9astJ9cvrs8zcJHMJUm5GJKFLdj1pyOB1v2xb0Cuk4RqcAUPinMhAwSgPXYtBjL45pwtvPOt5k8vEg0_dLCUeXCKR6hf7CkO9mLnPK0aQoB2IJQs7hFTJNCdttxlB_s9MeXMHrVZhk3kUdmEjbgP8zLWfoNfC4wT3jWjY)
19. [scribd.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF1TruNnjxfgr1bH0GEMtlrf9CpkTu4OVO_xpal_r5qdMIcRmOgA2zs7Ey0bUNWYP1LRmeZC3A7B28y_Yy_g7h902EDOf4IRQxXYdudUkV3xLNtKbgq5CBWxCnC_6Qj-kkwljtN-o4=)
20. [alldatasheet.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH04cKWdhYNgtGzCoqG8zplVdgLVTp1NbfiuVcBJovHRDYfWE5nZjXI-lhRMks9iRgh-4WFoW_DNeW9EWfhlFuterwT8Mg9tsNGva_uqflv9HN0XPhfuk6vlRSs8RxPNMOELK5EPYU0srw90e-Ok_1pzti9LRrexAS4hTp3)
21. [datasheet4u.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFTVUoq-whfVOVw3CqDKdykA41nxMUQEC95L3MoXu0j6i1Pxit2C43Qf-RqNtH5XpxL2kpReuBjKauIzblwOqeMlCfxp189Fhm8wXkSd5oqMHzH6zTWaDh5lNS-CFQMM2J79Ye5t-reflczMhFr5sdvyhdRIN6V7Zk=)
22. [ti.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEYRdxpvvFyHLse1bUduKBNOsgWHfa8CU-uGsWPyXLcl6qMPMTQy1_XFz6qY5lwkRSYRmH_afBERjRTf6FK1nPhfH6HPWO-I7bMlatZ5cY6ozOrtbgJtsj2nU01nvyZD-7TMA6tlXmcwsWHaY0sww4xk7IbCg90-32nkWxJVLejCZwx931-GHKpPHj0pKdeQXx2yLw5cs4ZareCWZ_AE5EJJqhUX59F_76oG9lFsublgsFiL_YSQhAjrU_t-B2Bq27tpcaQviEjCrPdOh6ldO36n9o=)
23. [analog.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHPlBQNoqqk9pXJcsB8x71pT02t4hf7C1YwWRxbi68rjoN2ARAuvL6f-6WbKoR1AvBlXoIEgauyJtj_A_Q_kaaXGlTI71pAMXwckvEB-FstgnDQTjM0Dc8L_SDodEYO0nybbnLv_JT200srKiLyZNfTqYTz3NspLjU=)
24. [edn.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFzmzxM1O0OJ51IJy8JuTxkuvva5r_1AD5Q6YDGvq_ahVgUmibZFS-wvXpsATWwY8ubYkvh_353r5MJmyJjrAAhQdhVOWTdPuVLdauzViUqxIdsYJBW-LYWih1Ruy8c1yREpMF5AMWZHngzAorF2qtaSucO_KcTd4Otch5Dn3NyxF8=)
25. [analog.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE3qAWRe9YuQyg5uHNBz9MhESZwG6tE5kdPGOy7Sf1Y1M24ecupkKG-VTQ4H_YZKOXvjYsZIOdPQJQMhOXIq2oxGiL6ZjoEvWTeXomYw87LdnvzAX4aeRZlUloB6R-3cqLjVPwQjA==)
26. [youtube.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEoH_rCtNa3I13erAl9r1BMgjqbb1uC_It9a5IMpudnamx9XOZr3nlaL11CxlSPfJE05dBa_VGH4MK0uIuV8gSofSe2EmdLZKYeITYmvnJzy4-vO7cgCTB7CP_CLc13Po0N)
27. [aliexpress.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGZNOL04M9Na7ZaArh4Z95CKqJc7IOqoaU0q4creadUAgQW1ELhH90BGG3kwGLPrELlrqN94iPKivt5Y8QVvxE-5Ts-O3bKU9r6y6-KfM4qw9HAhc0qVlLCzY3Xtu7stjYAlCgwvtXj9Q==)
28. [aliexpress.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHA2wlkI02viIqgnwWpBic-CcToqN6F0GntxnoAm7jCA7IWvh7sMgvIGuRAwTZlZr8WZ74IBD3cru01lUa0Zla8yXkHSRUC-htFBAPYiaK-UCZff5vhPtswJsWs4BIdOYGEJWTG3w==)
29. [weipuconnector.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdjIjglJBp923605dWrgBKovq1dI1Zycp1vjEO1c3u4Pr9IIvzBs6KKHTmn7g_QUmOKRl_cQ-gNW__4wPPcaPjoJUhA6FWyXKkUpIJgZioiPL5-bH181zHC1c-9iH_b6vl1Gk=)
30. [weipu-group.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEmkRA0sKkl0eKtFAp3_4cxLDCvgYUZ8HamURTu6MsjVGfajdPRG1Y6ZtkkXFMZFSDoSpcNlvnSrBtaIL37InS3VSAy1CT3cQ9sauMS_mgiUz-rBcwPUa0gDWcMChMKwQYScdKehg-JW_8abRRbakI_V5_i5PwbvKin7t_IZBzatKjjGl9Xnpa-lbcGUpLj1RiLrjT59eJDCkycjUuf1lMWgEjz7llf0A==)
31. [ebay.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGoIJrHHB7WY6lqmXd_jC9wK0Kkr3DZLLolz-ceeE7kkbC7JW3ZXLDIPhIrjfEcQcAgH5T4epLIDbaZbzvB7KD6TrIcMsv4lQAkKCoBRpIZ7_jbZtiRyIFOHz41)
32. [evelta.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHbrknlmXcRIArIzyOodvCsQgwYdLlaNF4YfyqDi3_foinFCxnYWCEfgMf0Fni7oespWMTV89jYBZX48l1uJWOYNNVLC3OFLRfd_vQ9Q2WWaZElEA1K2MD_qDUlWwPzKn5u7nAOcZOzp4BinEMws__OR37n_m7vRPtZtTyMDUrRv903llKK4MIMbzAvXRy0hOONAPzm6j5OhXLn0v2NMg==)


### Citations
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF5tqMgtAZgrG_E38srXlqRpMRPMSE-kErRMfEUMNoLE5_dJGMY-x4C-O8NLLFV7nBRN5WUv2l5r2HNBPId2WH6Cte2A3PFQB3xfKeCjsZv3RSRJoGhX0c9JAjLqYe-pHecfC5VUK8e4oJ4hCTvJwuYlFm7Wx14J1DIDo_9BofWcFAR60Ye-q5qdPieRSJMOr0=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHDa7yZemy3NDKNZ6DKer3461tFdmpNJlNLwsSIxQNF2Q9oqZAxoMp-coCRXp4rNO5O3O8AndmRTBiimwe6odIJ-T_lySVKLAhMQI5Fl1EYm1Y4dfjwvod5Bs-EJVnjaU4AYDA=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8gxoN9OKoaMaqB-VjnT1stbxdH4Om8Yp8t-sgjhFXkHkYTOw0q69YOBF2OStSa-2cW-YiOeqo1uCDSydTQ7VyCcTdHO9TbQgD6ylCE47cXGmrC8K9QDKGfsPt13odBetlKqrwa2y4K1fLkEmAvX3uJMZIUWiBpi53U121DoxCkcM6HGITlpNZl3Ie1GYLzmK0FI7tM_723mLoAw5exfS2R2p5K439r6nlRgj3p5s=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEnYLJl1gB0J53AvVCBRh1d4eKkQ_DbBId_6nHyDRdzB4yTP5hmm-Du2FUAHz1xS72ScVANg6mNjVhu0UoNS_Essa1yLXV9SXZvi2RYSZp2FkSIZdaXOTL_KRDQ
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGsU7MnlfzIC0IMnbDU80XtcUvOhSeYUU_hAht_GUIvbrVCCInYaDB5K3IvwXYe1wwdBW9jytx68Cc706ZupcANv_Kbbp7t8pTddS_rEDs5kocGor62-1QZzxYHVizb53EmytT764O-OI5EwSlnrLmv6s2BF9w7QQ==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEE75prjNlijqo32lg9OK_GkAvVZ0yDnVZy1pHUJQhPrf_PEGkSxazg-hFM5uxPj7z47bC0UVn4hX9T8p5l_4nr5Tb8LjclpeN5ovSrRtGz5dyXAOEClzIFMdNLauIZPZ4RLTqi0VaFB_1d-A0vC6ZneXq228BSoFI_CPIch5PUMVR0vXQ=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGUXsotge-QZBwmEla607wzf6U2nUNAijFmyFGX-A06erhCQnAzdEUoOSiUNB1D0Yd3FJiywcgHgd53gWpeWS0YiKbbIjKFHUuCU6JUZ4_kPecOg0OpMadEZ8p3An1I8u5s2jU=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0mzJW1aZHZjKEy-SvYlPaMxg-o6zIxEgH-2zH2DMAQMC1t4ky3LLsFT-7bKM4FtUM2B2aY6_MCOcrWgK0OS8QE9w6dyxbDteajkBAQTA70FqPWO2ij815ZiF6GjFxHEssooMXndLmafgUV7Di6p-rIINq6pVzK1tGGw==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCzZVBPXEBKMQBs9jmKzoI275FlYjwFjXOuA7nNhEEYXNgBdI0-MAqpN-_03mC1e7dkT9alTV2QnZ04b_4GIZVwzPTm5nOBxJsBLQzmyo4P3bV7Qe3EQ5NFRtigS3UXKESeTxL-0GvpUi6AbtMJQtNr3HoPgiVbg==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQET30Vsn8TrRwixBeurkBeb-VodTutH7wJzdiv1WYJUMei7jUZyBKsQcgwjsDl1Xw-3lI7BYTg29KABy5mUrseemXDmVnzmEHd5B_s4kH2fxHjTUz3siEkWusG-DN-tDTi6pOiFBhFNyQnYoMWhdicOBIIsQ-UZ_bE92ewBQJxQxDPufcVQSF29ZM9VOEfOxF61ZhoC
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGnHiZVeUaI8FXCSodKtHehmHqIQPYl20X8lWDPZmiXCNY_KNvTfZajopFPXPBtjbyyu6354qJP_TZl7E2k2JFbhE0T_RBH-Sd8Pbj4tVfMW99AWjgdCz0gyc7LrrHs3UbVvDRjlFcsE9Wt3nipWQ==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHtP2eTA9ej5P_p6aIYM5s2VbZfrCUHrc_6Ce6MMSYLJ3Gtccc-WOm0nZeT_Pib9DeN7UB9OKRh6kUfqvl9PpAh_DDsPSlzwMwZ8_Dqg7qStS1_xrfG5aNdFScYBihFJ2fLYsoDcRZ43krfyDgYj9yB0t7o8hAUBhTtfcRtBrVbDWoPnuU=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGqXF9wlmdypSCaIejPBdZTnf5XPgyjconnqvuPOik1dJ72ljw2oILaeOJBcTrb1G5WqZvNosJUCsbrN8YN3-BDyiplpAk8bDcFs65w5wqJBaCzzEiR9JmbsU56R_FlbZ2c1asB1Y9hypGW5UODm_IoojjxpsTeUZY-BvMR
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHAb2n2jcYtFPeHOq6ybDPE0rU9EYcgay0QhgqCWNB3a6rB7JwurbvD1RMEg5_J6Qw5Ugzsgu9KYsjMsydjRyq-W7b1zDfI75NRyAzIe_qgOcYBdJjiQNU=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGvcY99N14-CWxpfe6fNxSiZJOBSeqB_5u43F5GmoqcWIdR2zSKaTc5cgt_WBJReIp9DKqJS8y8nTGsbGaea9WORmtvSvohbysK6yfhHlkBERvEuGklsHfdKUuNxnh8xyS6XYAIWgdQOQ3y0yeDThmBTVY=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHnZTheHWz11cb2PjVC4CZifY8vAjIHHAup_S6JwiecyO_vYHt23-L5QUlg_Ga4I9OwOpp5-uuIUDqImQk7pPurCdZiiYKsMZJYEK422A4HtecFZHHc0YLFlSVkJgQCfq9LqmM=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE9astJ9cvrs8zcJHMJUm5GJKFLdj1pyOB1v2xb0Cuk4RqcAUPinMhAwSgPXYtBjL45pwtvPOt5k8vEg0_dLCUeXCKR6hf7CkO9mLnPK0aQoB2IJQs7hFTJNCdttxlB_s9MeXMHrVZhk3kUdmEjbgP8zLWfoNfC4wT3jWjY
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF1TruNnjxfgr1bH0GEMtlrf9CpkTu4OVO_xpal_r5qdMIcRmOgA2zs7Ey0bUNWYP1LRmeZC3A7B28y_Yy_g7h902EDOf4IRQxXYdudUkV3xLNtKbgq5CBWxCnC_6Qj-kkwljtN-o4=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH04cKWdhYNgtGzCoqG8zplVdgLVTp1NbfiuVcBJovHRDYfWE5nZjXI-lhRMks9iRgh-4WFoW_DNeW9EWfhlFuterwT8Mg9tsNGva_uqflv9HN0XPhfuk6vlRSs8RxPNMOELK5EPYU0srw90e-Ok_1pzti9LRrexAS4hTp3
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFTVUoq-whfVOVw3CqDKdykA41nxMUQEC95L3MoXu0j6i1Pxit2C43Qf-RqNtH5XpxL2kpReuBjKauIzblwOqeMlCfxp189Fhm8wXkSd5oqMHzH6zTWaDh5lNS-CFQMM2J79Ye5t-reflczMhFr5sdvyhdRIN6V7Zk=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEYRdxpvvFyHLse1bUduKBNOsgWHfa8CU-uGsWPyXLcl6qMPMTQy1_XFz6qY5lwkRSYRmH_afBERjRTf6FK1nPhfH6HPWO-I7bMlatZ5cY6ozOrtbgJtsj2nU01nvyZD-7TMA6tlXmcwsWHaY0sww4xk7IbCg90-32nkWxJVLejCZwx931-GHKpPHj0pKdeQXx2yLw5cs4ZareCWZ_AE5EJJqhUX59F_76oG9lFsublgsFiL_YSQhAjrU_t-B2Bq27tpcaQviEjCrPdOh6ldO36n9o=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFzmzxM1O0OJ51IJy8JuTxkuvva5r_1AD5Q6YDGvq_ahVgUmibZFS-wvXpsATWwY8ubYkvh_353r5MJmyJjrAAhQdhVOWTdPuVLdauzViUqxIdsYJBW-LYWih1Ruy8c1yREpMF5AMWZHngzAorF2qtaSucO_KcTd4Otch5Dn3NyxF8=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHPlBQNoqqk9pXJcsB8x71pT02t4hf7C1YwWRxbi68rjoN2ARAuvL6f-6WbKoR1AvBlXoIEgauyJtj_A_Q_kaaXGlTI71pAMXwckvEB-FstgnDQTjM0Dc8L_SDodEYO0nybbnLv_JT200srKiLyZNfTqYTz3NspLjU=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE3qAWRe9YuQyg5uHNBz9MhESZwG6tE5kdPGOy7Sf1Y1M24ecupkKG-VTQ4H_YZKOXvjYsZIOdPQJQMhOXIq2oxGiL6ZjoEvWTeXomYw87LdnvzAX4aeRZlUloB6R-3cqLjVPwQjA==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEoH_rCtNa3I13erAl9r1BMgjqbb1uC_It9a5IMpudnamx9XOZr3nlaL11CxlSPfJE05dBa_VGH4MK0uIuV8gSofSe2EmdLZKYeITYmvnJzy4-vO7cgCTB7CP_CLc13Po0N
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGZNOL04M9Na7ZaArh4Z95CKqJc7IOqoaU0q4creadUAgQW1ELhH90BGG3kwGLPrELlrqN94iPKivt5Y8QVvxE-5Ts-O3bKU9r6y6-KfM4qw9HAhc0qVlLCzY3Xtu7stjYAlCgwvtXj9Q==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHA2wlkI02viIqgnwWpBic-CcToqN6F0GntxnoAm7jCA7IWvh7sMgvIGuRAwTZlZr8WZ74IBD3cru01lUa0Zla8yXkHSRUC-htFBAPYiaK-UCZff5vhPtswJsWs4BIdOYGEJWTG3w==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdjIjglJBp923605dWrgBKovq1dI1Zycp1vjEO1c3u4Pr9IIvzBs6KKHTmn7g_QUmOKRl_cQ-gNW__4wPPcaPjoJUhA6FWyXKkUpIJgZioiPL5-bH181zHC1c-9iH_b6vl1Gk=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEmkRA0sKkl0eKtFAp3_4cxLDCvgYUZ8HamURTu6MsjVGfajdPRG1Y6ZtkkXFMZFSDoSpcNlvnSrBtaIL37InS3VSAy1CT3cQ9sauMS_mgiUz-rBcwPUa0gDWcMChMKwQYScdKehg-JW_8abRRbakI_V5_i5PwbvKin7t_IZBzatKjjGl9Xnpa-lbcGUpLj1RiLrjT59eJDCkycjUuf1lMWgEjz7llf0A==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGoIJrHHB7WY6lqmXd_jC9wK0Kkr3DZLLolz-ceeE7kkbC7JW3ZXLDIPhIrjfEcQcAgH5T4epLIDbaZbzvB7KD6TrIcMsv4lQAkKCoBRpIZ7_jbZtiRyIFOHz41
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHbrknlmXcRIArIzyOodvCsQgwYdLlaNF4YfyqDi3_foinFCxnYWCEfgMf0Fni7oespWMTV89jYBZX48l1uJWOYNNVLC3OFLRfd_vQ9Q2WWaZElEA1K2MD_qDUlWwPzKn5u7nAOcZOzp4BinEMws__OR37n_m7vRPtZtTyMDUrRv903llKK4MIMbzAvXRy0hOONAPzm6j5OhXLn0v2NMg==
