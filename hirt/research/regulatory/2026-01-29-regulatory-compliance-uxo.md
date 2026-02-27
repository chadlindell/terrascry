# Research Report

# Regulatory & Safety Compliance Brief: UXO Risk Management & Geophysical Survey

**Date:** October 26, 2025
**Subject:** Regulatory Compliance for Intrusive Geophysical Surveys (HIRT/MIT) in UXO Conflict Zones
**Jurisdiction:** UK (CIRIA C681/C785) & EU (EIA Directive/Valetta Treaty)

### Executive Summary

This brief outlines the regulatory and safety requirements for conducting intrusive archaeological and geophysical surveys in areas contaminated with Unexploded Ordnance (UXO), specifically World War II impact zones. It addresses the deployment of High-Resolution/Hybrid systems (referred to here as "HIRT") utilizing active Magnetic Induction Tomography (MIT) signals in the 2–50 kHz range.

**Key Findings:**
*   **Regulatory Status:** In the UK, **CIRIA C681** (2009) and its update **C785** (2019) constitute the de facto industry standard for managing UXO risks. While not statutory legislation, compliance is essential to meet legal obligations under the **CDM Regulations 2015** and the **Health and Safety at Work Act 1974**. In the EU, the **EIA Directive** and **Valetta Treaty** govern the protection of archaeological heritage and safety from major accidents, but specific UXO protocols are often devolved to national standards (e.g., German GEOSA).
*   **HIRT System Safety (2–50 kHz):** The operation of an active MIT signal in the 2–50 kHz range presents a **conditional safety risk** to specific types of magnetic influence fuzes. While standard WWII magnetic mines were typically designed to trigger on quasi-static magnetic signatures (simulating a passing ship), active electromagnetic emissions in the VLF/LF range fall under **HERO (Hazards of Electromagnetic Radiation to Ordnance)** classifications. High-intensity AC fields can induce currents in electro-explosive devices (EEDs) or anti-disturbance circuits. **Strict standoff distances** and HERO "Unsafe" protocols must be applied until the specific ordnance type is ruled out.
*   **Intrusive Survey Protocols:** Intrusive methods (e.g., CPT, probe insertion) in high-risk zones require a **"Look-Ahead" safety buffer**. This is not merely a static buffer but a dynamic operational procedure where a magnetometer sensor must scan 1.5–2.0 meters *ahead* of the intrusion tool to detect ferrous anomalies before contact.
*   **Permit to Dig:** A "Permit to Dig" in known WWII zones is strictly conditional upon the completion of a Detailed UXO Risk Assessment and, where necessary, the issuance of a **UXO Clearance Certificate** following a non-intrusive or intrusive survey.

---

## 1. Regulations Regarding "Intrusive Survey Methods" in Potential UXO Areas

### 1.1 UK Regulatory Framework (CIRIA C681 & C785)
In the United Kingdom, there is no specific legislation solely dedicated to UXO surveys; however, the **Construction Industry Research and Information Association (CIRIA)** guidelines are universally recognized by the Health and Safety Executive (HSE) as the benchmark for best practice [cite: 1, 2, 3].

*   **CIRIA C681 (Unexploded Ordnance: A Guide for the Construction Industry):** This document defines the risk management framework. For intrusive works (such as Cone Penetration Testing - CPT, boreholes, or piling) in potential UXO areas, the guidelines mandate a four-stage process [cite: 4, 5]:
    1.  **Preliminary Risk Assessment (PRA):** Initial screening of historical data.
    2.  **Detailed Risk Assessment (DRA):** In-depth analysis of bombing records, geology, and site history.
    3.  **Risk Mitigation Strategy:** If the risk is "Medium" or "High," mitigation is required. For intrusive works, this typically involves **Intrusive Magnetometer Surveying**.
    4.  **Implementation:** Execution of the survey and issuance of clearance certificates.

*   **Intrusive Survey Requirements:**
    *   **Methodology:** Intrusive surveys are required when non-intrusive surface methods (e.g., towed magnetometer arrays) cannot detect deep-buried ordnance due to depth (typically >3-4m) or surface "noise" (ferro-contamination) [cite: 6, 7].
    *   **Technology:** The standard method involves a **MagCone** or similar magnetometer probe pushed into the ground using a CPT rig. The system must be capable of detecting a 50kg German SC-50 bomb at a radius of approximately 2.0 meters from the probe [cite: 8, 9].
    *   **Operational Safety:** The survey must provide **real-time data** to the operator. "Blind" drilling or probing in high-risk zones is prohibited under ALARP (As Low As Reasonably Practicable) principles mandated by the Health and Safety at Work Act [cite: 6, 7].

### 1.2 EU Directives and Archaeological Guidelines
In the European Union, the regulatory landscape is a hybrid of pan-European heritage directives and national safety laws.

*   **EIA Directive (2011/92/EU & 2014/52/EU):** Major projects require an Environmental Impact Assessment (EIA). The 2014 amendment explicitly requires the assessment of **"risks of major accidents and/or disasters,"** which includes the risk of encountering UXO in conflict zones [cite: 10, 11, 12]. An EIAR (Environmental Impact Assessment Report) must demonstrate that the vulnerability of the project to UXO has been assessed and mitigated.
*   **Valetta Treaty (1992):** Formally the *European Convention on the Protection of the Archaeological Heritage*, this treaty mandates the preservation of archaeological assets. In conflict zones, this creates a dual obligation: protecting the heritage (archaeology) while ensuring safety from the "material remains of war" (UXO) [cite: 13, 14, 15].
*   **EAC Guidelines:** The *European Archaeological Council Guidelines for the Use of Geophysics in Archaeology* [cite: 16, 17] emphasize that survey methods must be chosen based on the site's specific conditions. While they focus on archaeological detection, they implicitly require adherence to national safety standards when operating in hazardous environments (e.g., conflict zones).

---

## 2. Magnetometer Safety Buffer for HIRT Probe Insertion

The "Magnetometer safety buffer" in the context of intrusive probing is technically referred to as the **Look-Ahead Capability** or **Clearance Radius**.

### 2.1 The "Look-Ahead" Requirement
When inserting a probe (such as a HIRT system or CPT magnetometer) into ground suspected of containing UXO, the sensor must be able to detect ferrous anomalies *before* the physical tip of the probe contacts them.

*   **Standard Buffer Distance:** Industry best practice (aligned with CIRIA C681) requires a magnetometer probe to detect a 50kg ferrous object at a distance of **1.5 to 2.0 meters** [cite: 8, 9].
*   **Operational Procedure:**
    1.  The probe is pushed hydraulically.
    2.  Data is monitored in real-time.
    3.  If the vertical magnetic gradient deviates beyond a set threshold (indicating a ferrous mass ahead or to the side), the operation is **aborted immediately**.
    4.  The "buffer" is effectively the detection range of the sensor. For a standard fluxgate gradiometer used in CPT, the system looks ahead into the undisturbed soil column.

### 2.2 Safety Buffer for Active Emitters (HIRT/MIT)
If the "HIRT" probe emits an active signal (MIT), an additional **Electromagnetic Safety Buffer** is required to prevent interference with the detonation mechanisms of potential ordnance (see Section 3).
*   **HERO Separation:** For active emitters, a separation distance is required between the transmitter and the potential ordnance. Since the ordnance location is unknown during insertion, the **transmitter power** must be limited such that the field strength at the probe tip (and immediate surrounding soil) does not exceed the **Maximum No-Fire Stimulus** for susceptible fuzes [cite: 18, 19].

---

## 3. Safety Critical Analysis: HIRT System Active MIT Signal (2–50 kHz) vs. Magnetic Fuses

**Critical Assessment:** The deployment of a 2–50 kHz active AC magnetic field in a UXO zone poses a **non-negligible safety risk** and requires a specific HERO (Hazards of Electromagnetic Radiation to Ordnance) assessment.

### 3.1 Characteristics of WWII Magnetic Fuses
Most WWII magnetic mines (e.g., German *GA*, *GB* series) and anti-handling fuses utilized **magnetic influence** mechanisms designed to detect the *quasi-static* magnetic signature of a passing ship or vehicle.
*   **Mechanism:** These typically employed **dip needles** or **induction coils** sensitive to low-frequency changes (0–10 Hz range) or static field distortions (typically 5–10 milligauss threshold) [cite: 20].
*   **Sensitivity:** They were designed to trigger on the *rate of change* or absolute threshold of the vertical magnetic component.

### 3.2 Interaction with 2–50 kHz Signals
While 2–50 kHz is significantly higher than the operational frequency of standard WWII magnetic influence mines (which targeted DC/low-frequency signatures), two critical risks exist:

1.  **HERO / Induced Current Risk (The Primary Threat):**
    *   **Eddy Current Induction:** Frequencies below 100 kHz are known to induce eddy currents in conductive materials [cite: 19]. If the UXO contains an **Electro-Explosive Device (EED)** (e.g., an electric bridge-wire detonator found in some German and Allied bombs), the AC magnetic field can induce a current in the firing circuit.
    *   **Resonance:** If the firing circuit wiring acts as an antenna resonant to the 2–50 kHz wavelength (or harmonics), the induced current could exceed the "No-Fire" threshold, causing detonation [cite: 18, 21].
    *   **Regulation:** US Navy and NATO HERO standards treat frequencies <100 kHz by limiting the magnetic field intensity to ensure induced currents remain 16.5 dB below the Maximum No-Fire Current (MNFC) of the most susceptible ordnance [cite: 19].

2.  **Influence Fuzes & Anti-Sweep Devices:**
    *   Some advanced WWII mines were equipped with **anti-sweep** devices designed to detonate if they detected magnetic signatures typical of mine-sweeping gear. While most sweepers used DC or low-frequency pulses, high-intensity AC fields could theoretically interact with the coil mechanisms or anti-disturbance relays, causing unpredictable behavior.
    *   **Modern Electronic Fuzes:** If the area contains *post-WWII* ordnance (modern conflict zones), electronic fuzes are highly susceptible to EMI in the kHz range [cite: 22].

### 3.3 Conclusion on HIRT Safety
**The HIRT system's 2–50 kHz signal DOES have the potential to trigger specific magnetic or electric fuzes.**
*   **Requirement:** You must treat the probe as a **HERO UNSAFE** emitter until a specific technical assessment proves the field strength at the probe tip is below the safety threshold for the specific ordnance types expected (e.g., German electric fuzes EL.A.Z. 17).
*   **Mitigation:** Use passive magnetometry *first* to clear the probe path. Only activate the MIT (active signal) once the immediate volume is confirmed clear of large ferrous mass, or ensure the MIT power output is intrinsically safe (below EED induction thresholds).

---

## 4. "Permit to Dig" Requirements for Research in WWII Impact Zones

The "Permit to Dig" is the administrative control measure that authorizes ground disturbance. In known WWII impact zones, this permit is conditional on strict UXO risk mitigation.

### 4.1 The Workflow (CIRIA C681 Compliant)

1.  **Pre-Requisite: Detailed UXO Risk Assessment (DRA)**
    *   Before applying for a permit, a DRA must be completed by a competent UXO specialist. This document defines the risk level (Low, Medium, High) and specifies the required mitigation [cite: 2, 4, 23].

2.  **Mitigation / Survey Execution**
    *   **Non-Intrusive:** For shallow works, a surface magnetometer survey is conducted.
    *   **Intrusive:** For deep works (piling, boreholes), an intrusive magnetometer survey (MagCone) is performed at the specific location of the dig [cite: 7, 24].

3.  **UXO Clearance Certificate / ALARP Certificate**
    *   The survey data is analyzed by a geophysicist. If no UXO is found, a **Clearance Certificate** is issued for that specific coordinate (or site box) [cite: 4, 9].
    *   *Note:* The certificate is valid only for the surveyed column/area.

4.  **Permit Issuance**
    *   The Site Manager / Principal Contractor issues the "Permit to Dig" only upon receipt of the Clearance Certificate.
    *   **Permit Conditions:** The permit must explicitly state:
        *   "UXO Clearance Certified to depth X meters."
        *   "Watching Brief required" (if residual risk exists).
        *   "Stop Work" procedures in case of unexpected discovery [cite: 25, 26].

5.  **On-Site Supervision (Watching Brief)**
    *   For high-risk zones, even with a permit, an EOD (Explosive Ordnance Disposal) Engineer may be required to be physically present during the excavation to monitor for "blind" items or non-ferrous ordnance not detected by magnetometers [cite: 27, 28].

### 4.2 Summary Checklist for Researchers
*    **Desk Study:** Confirm WWII history (bombing density).
*    **Risk Assessment:** Commission a CIRIA-compliant DRA.
*    **Survey:** Conduct appropriate geophysical survey (Passive Magnetometry is standard; Active MIT requires HERO check).
*    **Certification:** Obtain UXO Clearance Certificate for specific probe locations.
*    **Permit:** Attach Certificate to Permit to Dig application.
*    **Briefing:** All site personnel must receive UXO Safety Awareness briefings [cite: 27].

### Table 1: Summary of Survey Method Regulations

| Feature | Passive Magnetometer (Standard) | Active HIRT/MIT (2–50 kHz) |
| :--- | :--- | :--- |
| **Primary Use** | Detection of ferrous mass (Iron/Steel) | Conductivity mapping / Discrimination |
| **Regulatory Risk** | Low (Passive sensor) | **High (Active Emitter)** |
| **Safety Constraint** | Look-ahead buffer (2m) | **HERO Safety Distance** + Look-ahead |
| **Fuse Trigger Risk** | Negligible | **Possible** (Induced currents in EEDs) |
| **Clearance Status** | Accepted for UXO sign-off | Supplementary (requires safety proof) |

**Disclaimer:** This brief provides a summary of compliance requirements based on current literature (CIRIA C681, EU Directives). It does not constitute legal advice. A qualified UXO consultant and EOD safety officer must be engaged for specific project planning in conflict zones.

**Sources:**
1. [zeticauxo.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFhXINVbRCnU3084GZYhoqwhgrdfIpGf6BRF9P4SHfeX3-bk6h64pm3y3g6Jq-hL91VZwAZiHAHIv_7FkiGoYKRGjAIWtRm2OL43hWjtxeFM1wZwcgor97ycXx-HKgm22Iy8qqmcyae)
2. [igne.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGp7eqco3GEr7NWVqAEhv1vDRRjTHspHItTLOxAdlorgVdBiojsKtb59nIBhWsxvO94WUb8PwEgfsqWqnrE35miqo6hDCEl5QheykQpqk3Oe5XeqfL6oXdGhuAhCRYpTjL7a4kTcJo-htBDWFiDwXGPKpRTJ4gj5ckUhyUjyyA=)
3. [1stlinedefence.co.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGxW1hXFlYQBvc2AcG_femtPr9vGmcZ2Qy3iX7G1eiXCz-dRNjbVc1VkRPjI-_Cgsjdpboo--w6GHtif7ZXs6OzbLr0iFtvtB9PZ8xLcYd_lSLaiuustx3L3JTRNXYjPsOpPyajjyH8Zi2r_z-WzB69SPQP3s0w-VDyjX0nXKcGRlWl-yssRk8AThIa3BcO0VJeXAMq_-XjjXE0etvgjig9I8w0UTdM9M0ZcTFVK60=)
4. [siuxo.co.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHhseB6Yx2bVqckXqub4_M2bxx9rhV9-_cu2AQM3Z-7cG_Gh26ZjD3Tfu5SaooQIljsjRoW-DI8AZ09DOT8DZX9D82BPWprpJpIYzpZhcojbcf7pz8=)
5. [brimstoneuxo.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE2H6OA5pUa17ojoVl30dveTwAtFuKUsrOI0YfZvCKlGf6_0YJLvSA73QubjXFYSGlkSclGjxKuNdPAkaT4u3WkVVJqA3yP6J-GpSfBto51gmjUaA14O9DxTxiE-aYGbdEynyN65bwEP5uEG-NIA3QyJf3i8hc=)
6. [brimstoneuxo.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFYWbIYNMVkW-zWyjZeSKQnWx3GE8gTxaOol8A-F2ZGBVjPa9p3JBMX8WpcrH_qkeSfe5W5sLIJUvOAhiBZzE4BrbWIFElybSRsxlSUwHxhp8ycNBm_EGAOekxr9ek4XUHZFc0KtCz_sqKGlHTC9anjBTPm)
7. [1stlinedefence.co.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEsolNNejShuCl67pgyoOfop8u0WiNTG9kPm8zRWLdaLg2TBXIq8oxLuRUwa8Vzpt3-0d-GyQEgPIfvVnhdmQ3aKiJNiNBdg2SnDDDRQhDNreH2ZvzqN-1-HzEqBXa-H3f1Oked-e50zOPay8JDe3JDJBefQaFdvy0S2EpQfTe2LEjM1ObY1eSoOQ==)
8. [insitusi.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEY92sM-vAzHVy6Lvt1Hrr4ZFr8LqfY-aXCVDts7j9k_6wPZLUVgCQ7LMkJuxJFIjDeOiokgTTPgJcqr5BF5Yq80wPoLQGKUtrUH_sGP9QpthNIecxylXWwL85-suvLw60R3TStbOGcJSpM3VhYqgW5AGgsTR2syp-1mn-JXh1YOHO2ak82waBdr7xoohhcLtfksVd6T1y1Hg==)
9. [socotec.co.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF4n4sZRZ21xrVuult-u9wc-ZnPNSAst7nTMTU9fpEoETkqA4v8NlElfUkSC8cj9T32anvbxnjHJJh2rWeR6wb14rfisH1Q0ml6wGpZQo7OvEt0GyT8SYxrdp32yqoabZMxcjILMHo-PoiVSf8BCHO1QV8QCtC1qnh79U90BAck0Wypx1Z_hFLTLX6EtZAXaQ==)
10. [pleanala.ie](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFiaZ894vbqjjEoNFXjM-WRX88LR3cBAcCpmVprvWh6iwNrAQy4MIhR7A-HM-KLPpRGOjURFFRdRYyS5XLrq7sqiURl-VSZTGSy6mqr24mgwRWUeP6HK_p83ntmdViAU-7KtwO-fxJrzRjeKCwNeucvlQhyJbHRgUDVrGiv6jicJI9qgVbR-uRvkAKwb2jsCM38yFkoY7FS454YBB59t6N3JYw0K_0NTsgdlXI7VsQ-ecFuNXcmGuSElSYz8c8awl4s6OeuBI12IwOj2xJps2_h2M36_A2LqCe81v43Is7zkuUwFs2oN63sSYu22mM4i72mXrnmLlC8LSCtm5bFAtBkBIs=)
11. [gov.scot](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHwTQa6dJ_qzN5N2NnKYJwml-YzDuCjlqvFRfJgGFgAWYKsxQnS4EoyHt9Hx86-JRgu9VU-WJFll4afPoCFevv6QAxbB6HSvSaEvaVl1gBFpNg85r0VC9R02pvEW2ax1C6Gfe4DmnIcCq6PrfJOOcc7eWS0uNbvvAVkC5ed)
12. [dublinarray-marineplanning.ie](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdICyuAEUgub-O5vKVCip6ssVDgt1K7xdYGEPt6sd2XDQtWBQVVSlGrqsPPJ1SRVwv8CyzJ6xeBLDvkLJ3Zck0shI05Fx0SscTCxnGCQgTmTUY1uk5YaejE65EXzmfCx_PjeBuuQObRJdu8tSdGT-zGxFXAwoBvdn1IUqvva24YD7X97tAzsD0bd0vQ7XJTg2ZTr0qy86qaII8jJG-QMTJWmVryM0hGzQB6ZXovkM=)
13. [unesco.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8C-0j3vN94uqidy5Q4rMKhIbpBsNyPxtRx2OEn42hTgak0jv36bpt1M9hIHhQFWWcfWIrZU5Z5NNGDdkJggTIOeLw0jpKi4H1z7xVAzmMFDxrGpGDJNJUo6hdTA==)
14. [sidestone.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0o6PPNEThF2361-zSRq3Y4aZ1nTgIHn_ejuV1pmkQXd5afLPBDw0nVSZJuTj_THlsFRZHlJlXybE_hp7Y_uyn23epPowgO8muSoC_V3YgNaj0s-COHniXYn8_MF9uCR5tvuWBlQ2Bfw4JoN4=)
15. [conservation-wiki.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH8BmVXppguq6yX3HE7kJQFS5_obqsT8f9eSBNhsa-78JRyLE3n8shcbR9ERmrpTvM_s87CH8SIxbGsBqUxJx_j-MrYebKovtN21Jh3AESkJCwfaMlcglmqTHLqVRa6aRvbjkoP0hGg33MHjLakevL9kX-SCfKk8aV4skSNRf7m1f3p3Vcxk3mCqlUAtwFm)
16. [archaeolingua.hu](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCKLDRaMH6ZaIdHzpVny7DsGeMGC1C_2_k-9HBvWUC02w2YFfRuhftfIF3dOv7N5BZFnBKi7DJexVA1RdGzZwFX7gh84pewx4bGllxOI4cYGxo_3SOIMF9E88Hee6Ak2WwOXxLXr6uw179TVIzszYp4UsQCzZOQK_orX3Xj7w-ng==)
17. [historicengland.org.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHvHJAc-BKQbKaAKLF2PSN4jygYwrfB88V7RlBCM-vtMWAqNidO7QnuN_N5ipt3rT8l68dKLGSotzBf3DXnIgRHM63TMQS9e_vv1JY3R6rf9ce13S-KCl7bksLEo-iO86wJcSJKdP6uQ2dPY9AeSU0yn63G7L1YVprC8UvJrolaa7-Adq-1Xe-qhN4VdxJtsnI0qm4UQ3PkgIYLUH6zstQhdqPC)
18. [service.gov.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFpdC-3LhTMYjSEzBeQeBYTmhIylHAEI5oRbVkpT2fIEOG1OOamfz4xMbu0_SfQjcGLgUl1Z284tzlHAnqd6I9E7NEENHEtp_Nf0Xe5RuZSSvB4e6Ymt2vnkbBSNxcogbt5rlIBX3eKpiZDuSajM1qm-GAfCCT33tJWE8wwNc41p0n74uiODN9SfJ5vXqDQalKg_TaArrPzQFdgHfDcasw=)
19. [dtic.mil](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH_HxEIM5SrLtP9yT7PYSlgS8WCUmzPLNIYVTk40IE3hfzTTNgZLT5R0B-vC-3Vr5PDwu-JH7pWgbcUL_5Df07EluhXyOxW38XltO5Oa5lYGmfL7icBdpZ30UE_p53pxmBnMF4E)
20. [researchgate.net](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEIooCrEGSDS-6FFSXckFJS_n4T1-8LSuSKBQf_v2bZJdhDOxIN7XlHT8dtxzBJ4WTMiwSYAPsiL8cBp0TBpCkB34o8GX9nnCk42wrze2sGsptQxAy362Z2PLcjYGWYTeYduqUt2HT-NTtFQm87Y3yb7Z0D0k_3VoMGFWaTnhIKWzEvGguM96Ez7uPrRo0=)
21. [af.mil](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHSB89voBR8ZvNAhK4jUUJlXiOEsxoQGE9AiRPZcV0NblCp5f-tvXSpVAVMj5X-45GrKU79gpFlgshtrNHtZaqazMNAWDGuqCPi3nLX0tqgsILRNT7EzGLoamDWgffcz3ffAwFgElWiqAGxORsPQFVmCjvtVEqbWWMkmYc0JG0dZiK0J5iawrlk7vQNp0z0Elx2)
22. [unsaferguard.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsnWk8oUkbtt9orTt_a5WjBp62SYd0wj6oVgkl8-H6-6WiPy-rHG6OQIzNQu8TXIdGGmDr09Zc3ZZi9z68eMFiUxZON_mUR7LKPP1mPBei5xzTtZKRJDRTvrcR3IZadVIldOcNe0jITy9quzIS7s1U7Q1Kh9Gsoi6TPSYZ5OLzauo9A5oVjtZCtZbddPv0fUyVuQ01Pw==)
23. [brimstoneuxo.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFJzqe52Lj_dE1E7n8qX9nSjXRv9T15FmSQplh08yjjl7EQlLT0Wy9ijSgoueagpe__ED8ROtl95tD7Dwujess37pMwEDDOI83BvY60312_5m2BVDcrOEQ7zYZa5IFDkwE_EYDM7Ny2oBda9QNg7Vbrp_2yeURdvaOCfCE=)
24. [brimstoneuxo.com](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFYhNqSkcHbqIKj86mFiuUbaTdCU3GxEc5J349J53kqQfQKncGKRe24IE4xeX8miiLHDeubzZ2PFvd4ibHlsnoaO0W_5arMTWisXJAFTKkzF1x4kADJnwxuSw7f9lKatI_eyB7dFhXn5R9eT9j-TrDtAoV6N8w_Alo=)
25. [coventry.gov.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF2e_VfN5fUUX66h4FE4iR5gUqjU-rYu76f2gSQFgaqrzsyJd20n1a3tgoyaKf9j-fQXtt8718w72OiMPaqOz2D0G7O-8M8I4Z4cqNIqfWzjykChweoPt_LN9UtUEFH25dNI-wVOpjsXa7Ho7WjMHAMATNYbFfAhq_sl62LDTahXAC51A==)
26. [citb.co.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGdxEEScrOTtd9ZvTYRuQHiZq3nCYPE2H68nYjxvwOv2pddLxMB6uAF131E14GZCK921MhY8JQogrlBPEBLUBwwVy6x0EFkDNrioxQZCznJefg3lk4Yn2BDeJPCz9AMaF7c6Ibplgx3Wg_fIA0RXUg=)
27. [1stlinedefence.co.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFzbAsc4Wz5Ys5nrZsEBx7pffTMsQyBDuFiioK8aEezYn3kfAD3p4jI-dG-KkkAy1hA7fr85Ze_0Jmbhq4sLb8II82QY8HsYxdlCZSo5_w84sMp412BB9F179LQzBAxqt3xn044UBK2vo93yxsarAgXak0l_J45YPS0OX-wl9jRoLWtGajx3uhcIy5_f4-fBQ==)
28. [1stlinedefence.co.uk](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHgvTjQNShm62xvTH6fyr-TSQwtG1bWX4vJCz-vh4PMxzl_dw8Egm6J-e-HDxLU4mGvIEQLw8lWtVpyBcwA3SzSQqehljcGn2EbdwwgPaYoM6t9IciLBZDHAIjOga68sBGqpj4KIOQdJBvOdZOtPf1CoO4Wz2Htk6lvbUJcEVs7xiEBOF0MSD9lUBT12JEVoA==)


### Citations
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFhXINVbRCnU3084GZYhoqwhgrdfIpGf6BRF9P4SHfeX3-bk6h64pm3y3g6Jq-hL91VZwAZiHAHIv_7FkiGoYKRGjAIWtRm2OL43hWjtxeFM1wZwcgor97ycXx-HKgm22Iy8qqmcyae
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGp7eqco3GEr7NWVqAEhv1vDRRjTHspHItTLOxAdlorgVdBiojsKtb59nIBhWsxvO94WUb8PwEgfsqWqnrE35miqo6hDCEl5QheykQpqk3Oe5XeqfL6oXdGhuAhCRYpTjL7a4kTcJo-htBDWFiDwXGPKpRTJ4gj5ckUhyUjyyA=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGxW1hXFlYQBvc2AcG_femtPr9vGmcZ2Qy3iX7G1eiXCz-dRNjbVc1VkRPjI-_Cgsjdpboo--w6GHtif7ZXs6OzbLr0iFtvtB9PZ8xLcYd_lSLaiuustx3L3JTRNXYjPsOpPyajjyH8Zi2r_z-WzB69SPQP3s0w-VDyjX0nXKcGRlWl-yssRk8AThIa3BcO0VJeXAMq_-XjjXE0etvgjig9I8w0UTdM9M0ZcTFVK60=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE2H6OA5pUa17ojoVl30dveTwAtFuKUsrOI0YfZvCKlGf6_0YJLvSA73QubjXFYSGlkSclGjxKuNdPAkaT4u3WkVVJqA3yP6J-GpSfBto51gmjUaA14O9DxTxiE-aYGbdEynyN65bwEP5uEG-NIA3QyJf3i8hc=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHhseB6Yx2bVqckXqub4_M2bxx9rhV9-_cu2AQM3Z-7cG_Gh26ZjD3Tfu5SaooQIljsjRoW-DI8AZ09DOT8DZX9D82BPWprpJpIYzpZhcojbcf7pz8=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEsolNNejShuCl67pgyoOfop8u0WiNTG9kPm8zRWLdaLg2TBXIq8oxLuRUwa8Vzpt3-0d-GyQEgPIfvVnhdmQ3aKiJNiNBdg2SnDDDRQhDNreH2ZvzqN-1-HzEqBXa-H3f1Oked-e50zOPay8JDe3JDJBefQaFdvy0S2EpQfTe2LEjM1ObY1eSoOQ==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFYWbIYNMVkW-zWyjZeSKQnWx3GE8gTxaOol8A-F2ZGBVjPa9p3JBMX8WpcrH_qkeSfe5W5sLIJUvOAhiBZzE4BrbWIFElybSRsxlSUwHxhp8ycNBm_EGAOekxr9ek4XUHZFc0KtCz_sqKGlHTC9anjBTPm
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEY92sM-vAzHVy6Lvt1Hrr4ZFr8LqfY-aXCVDts7j9k_6wPZLUVgCQ7LMkJuxJFIjDeOiokgTTPgJcqr5BF5Yq80wPoLQGKUtrUH_sGP9QpthNIecxylXWwL85-suvLw60R3TStbOGcJSpM3VhYqgW5AGgsTR2syp-1mn-JXh1YOHO2ak82waBdr7xoohhcLtfksVd6T1y1Hg==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF4n4sZRZ21xrVuult-u9wc-ZnPNSAst7nTMTU9fpEoETkqA4v8NlElfUkSC8cj9T32anvbxnjHJJh2rWeR6wb14rfisH1Q0ml6wGpZQo7OvEt0GyT8SYxrdp32yqoabZMxcjILMHo-PoiVSf8BCHO1QV8QCtC1qnh79U90BAck0Wypx1Z_hFLTLX6EtZAXaQ==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdICyuAEUgub-O5vKVCip6ssVDgt1K7xdYGEPt6sd2XDQtWBQVVSlGrqsPPJ1SRVwv8CyzJ6xeBLDvkLJ3Zck0shI05Fx0SscTCxnGCQgTmTUY1uk5YaejE65EXzmfCx_PjeBuuQObRJdu8tSdGT-zGxFXAwoBvdn1IUqvva24YD7X97tAzsD0bd0vQ7XJTg2ZTr0qy86qaII8jJG-QMTJWmVryM0hGzQB6ZXovkM=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFiaZ894vbqjjEoNFXjM-WRX88LR3cBAcCpmVprvWh6iwNrAQy4MIhR7A-HM-KLPpRGOjURFFRdRYyS5XLrq7sqiURl-VSZTGSy6mqr24mgwRWUeP6HK_p83ntmdViAU-7KtwO-fxJrzRjeKCwNeucvlQhyJbHRgUDVrGiv6jicJI9qgVbR-uRvkAKwb2jsCM38yFkoY7FS454YBB59t6N3JYw0K_0NTsgdlXI7VsQ-ecFuNXcmGuSElSYz8c8awl4s6OeuBI12IwOj2xJps2_h2M36_A2LqCe81v43Is7zkuUwFs2oN63sSYu22mM4i72mXrnmLlC8LSCtm5bFAtBkBIs=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHwTQa6dJ_qzN5N2NnKYJwml-YzDuCjlqvFRfJgGFgAWYKsxQnS4EoyHt9Hx86-JRgu9VU-WJFll4afPoCFevv6QAxbB6HSvSaEvaVl1gBFpNg85r0VC9R02pvEW2ax1C6Gfe4DmnIcCq6PrfJOOcc7eWS0uNbvvAVkC5ed
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0o6PPNEThF2361-zSRq3Y4aZ1nTgIHn_ejuV1pmkQXd5afLPBDw0nVSZJuTj_THlsFRZHlJlXybE_hp7Y_uyn23epPowgO8muSoC_V3YgNaj0s-COHniXYn8_MF9uCR5tvuWBlQ2Bfw4JoN4=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8C-0j3vN94uqidy5Q4rMKhIbpBsNyPxtRx2OEn42hTgak0jv36bpt1M9hIHhQFWWcfWIrZU5Z5NNGDdkJggTIOeLw0jpKi4H1z7xVAzmMFDxrGpGDJNJUo6hdTA==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH8BmVXppguq6yX3HE7kJQFS5_obqsT8f9eSBNhsa-78JRyLE3n8shcbR9ERmrpTvM_s87CH8SIxbGsBqUxJx_j-MrYebKovtN21Jh3AESkJCwfaMlcglmqTHLqVRa6aRvbjkoP0hGg33MHjLakevL9kX-SCfKk8aV4skSNRf7m1f3p3Vcxk3mCqlUAtwFm
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCKLDRaMH6ZaIdHzpVny7DsGeMGC1C_2_k-9HBvWUC02w2YFfRuhftfIF3dOv7N5BZFnBKi7DJexVA1RdGzZwFX7gh84pewx4bGllxOI4cYGxo_3SOIMF9E88Hee6Ak2WwOXxLXr6uw179TVIzszYp4UsQCzZOQK_orX3Xj7w-ng==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHvHJAc-BKQbKaAKLF2PSN4jygYwrfB88V7RlBCM-vtMWAqNidO7QnuN_N5ipt3rT8l68dKLGSotzBf3DXnIgRHM63TMQS9e_vv1JY3R6rf9ce13S-KCl7bksLEo-iO86wJcSJKdP6uQ2dPY9AeSU0yn63G7L1YVprC8UvJrolaa7-Adq-1Xe-qhN4VdxJtsnI0qm4UQ3PkgIYLUH6zstQhdqPC
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFpdC-3LhTMYjSEzBeQeBYTmhIylHAEI5oRbVkpT2fIEOG1OOamfz4xMbu0_SfQjcGLgUl1Z284tzlHAnqd6I9E7NEENHEtp_Nf0Xe5RuZSSvB4e6Ymt2vnkbBSNxcogbt5rlIBX3eKpiZDuSajM1qm-GAfCCT33tJWE8wwNc41p0n74uiODN9SfJ5vXqDQalKg_TaArrPzQFdgHfDcasw=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH_HxEIM5SrLtP9yT7PYSlgS8WCUmzPLNIYVTk40IE3hfzTTNgZLT5R0B-vC-3Vr5PDwu-JH7pWgbcUL_5Df07EluhXyOxW38XltO5Oa5lYGmfL7icBdpZ30UE_p53pxmBnMF4E
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEIooCrEGSDS-6FFSXckFJS_n4T1-8LSuSKBQf_v2bZJdhDOxIN7XlHT8dtxzBJ4WTMiwSYAPsiL8cBp0TBpCkB34o8GX9nnCk42wrze2sGsptQxAy362Z2PLcjYGWYTeYduqUt2HT-NTtFQm87Y3yb7Z0D0k_3VoMGFWaTnhIKWzEvGguM96Ez7uPrRo0=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHSB89voBR8ZvNAhK4jUUJlXiOEsxoQGE9AiRPZcV0NblCp5f-tvXSpVAVMj5X-45GrKU79gpFlgshtrNHtZaqazMNAWDGuqCPi3nLX0tqgsILRNT7EzGLoamDWgffcz3ffAwFgElWiqAGxORsPQFVmCjvtVEqbWWMkmYc0JG0dZiK0J5iawrlk7vQNp0z0Elx2
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsnWk8oUkbtt9orTt_a5WjBp62SYd0wj6oVgkl8-H6-6WiPy-rHG6OQIzNQu8TXIdGGmDr09Zc3ZZi9z68eMFiUxZON_mUR7LKPP1mPBei5xzTtZKRJDRTvrcR3IZadVIldOcNe0jITy9quzIS7s1U7Q1Kh9Gsoi6TPSYZ5OLzauo9A5oVjtZCtZbddPv0fUyVuQ01Pw==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFJzqe52Lj_dE1E7n8qX9nSjXRv9T15FmSQplh08yjjl7EQlLT0Wy9ijSgoueagpe__ED8ROtl95tD7Dwujess37pMwEDDOI83BvY60312_5m2BVDcrOEQ7zYZa5IFDkwE_EYDM7Ny2oBda9QNg7Vbrp_2yeURdvaOCfCE=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFYhNqSkcHbqIKj86mFiuUbaTdCU3GxEc5J349J53kqQfQKncGKRe24IE4xeX8miiLHDeubzZ2PFvd4ibHlsnoaO0W_5arMTWisXJAFTKkzF1x4kADJnwxuSw7f9lKatI_eyB7dFhXn5R9eT9j-TrDtAoV6N8w_Alo=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF2e_VfN5fUUX66h4FE4iR5gUqjU-rYu76f2gSQFgaqrzsyJd20n1a3tgoyaKf9j-fQXtt8718w72OiMPaqOz2D0G7O-8M8I4Z4cqNIqfWzjykChweoPt_LN9UtUEFH25dNI-wVOpjsXa7Ho7WjMHAMATNYbFfAhq_sl62LDTahXAC51A==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGdxEEScrOTtd9ZvTYRuQHiZq3nCYPE2H68nYjxvwOv2pddLxMB6uAF131E14GZCK921MhY8JQogrlBPEBLUBwwVy6x0EFkDNrioxQZCznJefg3lk4Yn2BDeJPCz9AMaF7c6Ibplgx3Wg_fIA0RXUg=
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFzbAsc4Wz5Ys5nrZsEBx7pffTMsQyBDuFiioK8aEezYn3kfAD3p4jI-dG-KkkAy1hA7fr85Ze_0Jmbhq4sLb8II82QY8HsYxdlCZSo5_w84sMp412BB9F179LQzBAxqt3xn044UBK2vo93yxsarAgXak0l_J45YPS0OX-wl9jRoLWtGajx3uhcIy5_f4-fBQ==
- https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHgvTjQNShm62xvTH6fyr-TSQwtG1bWX4vJCz-vh4PMxzl_dw8Egm6J-e-HDxLU4mGvIEQLw8lWtVpyBcwA3SzSQqehljcGn2EbdwwgPaYoM6t9IciLBZDHAIjOga68sBGqpj4KIOQdJBvOdZOtPf1CoO4Wz2Htk6lvbUJcEVs7xiEBOF0MSD9lUBT12JEVoA==
