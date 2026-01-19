# HIRT Probe Insertion Methods: Comprehensive Summary and Recommendations

**Document Type:** Research Summary and Recommendations
**Application:** HIRT Geophysical Survey System
**Date:** 2026-01-19
**Status:** Complete

---

## Executive Summary

This document consolidates findings from comprehensive research into methods for creating 12-16mm diameter boreholes to 2-3m depth for HIRT probe insertion at potential UXO (Unexploded Ordnance) sites, particularly WWII bomb craters.

### Key Requirements

| Requirement | Specification |
|-------------|---------------|
| Hole diameter | 12-16mm (18-20mm with clearance) |
| Depth | 2-3 meters |
| Holes per survey | 20-50 per grid |
| UXO Safety | NO percussion, NO vibration |
| Grid size | Typically 10x10m |
| Soil types | Disturbed crater fill, mixed soils |

### Methods Evaluated

Five candidate methods were evaluated in detail through parallel feasibility studies:

| Method | Feasibility | UXO Safety | Equipment Cost | Per-Hole Cost | Production Rate |
|--------|-------------|------------|----------------|---------------|-----------------|
| **Hydraulic Static Push** | EXCELLENT | SAFE | $50-200K | $50-90 | 20-50 holes/day |
| **Water Jet Systems** | EXCELLENT | SAFE | $500-5K | $3-10 | 20-40 holes/day |
| **Robotic Deployment** | GOOD | SAFE | $60-255K | Variable | 15-30 holes/day |
| **Biomimetic Root Growth** | NOT VIABLE | SAFE | N/A | N/A | Too slow |
| **Lightweight Purpose-Built Rig** | EXCELLENT | SAFE | $5-15K | $20-40 | 25-40 holes/day |

---

## Recommended Approach: Tiered Implementation

### Tier 1: Immediate Deployment (0-3 months)

**Primary Method: Water-Assisted Static Push**

Combines water jetting to create pilot holes with static push for probe insertion.

| Aspect | Specification |
|--------|---------------|
| Equipment | Hand-operated lance + portable pump |
| Cost | $1,500-2,500 complete system |
| Training | 1-2 hours |
| Crew | 2 persons |
| UXO Safety | SAFE - no percussion, no vibration |

**Equipment List:**
- Honda WH20 or equivalent pump (~$1,000)
- 3.5m stainless steel jetting lance (~$150)
- Fire hose 25m with fittings (~$150)
- 1,000L IBC water tank (~$150)
- Bentonite slurry supplies (~$50)

**Why This First:**
- Lowest equipment cost
- Field-serviceable with common components
- Proven technique (well drilling, ground rods)
- Can be developed and tested immediately

### Tier 2: Standard Operations (3-12 months)

**Primary Method: Lightweight Hydraulic Push Rig**

Purpose-built portable system for efficient probe insertion.

| Aspect | Specification |
|--------|---------------|
| Equipment | Ground-anchored frame with hydraulic cylinder |
| Push force | 8-10 kN (adequate for most soils) |
| Weight | 75-100 kg total system |
| Cost | $8,000-15,000 |
| Crew | 2 persons |
| Production | 25-40 holes/day |

**System Components:**
- Aluminum frame with screw anchor mounts
- 10 kN hydraulic cylinder (1.5m stroke)
- 4x helical ground anchors
- Battery-powered hydraulic pump
- 50m tethered control (UXO safety)

**Why This Next:**
- Optimal balance of capability and cost
- Portable (2-person carry)
- Adequate force for crater fill soils
- Tethered control enables UXO standoff

### Tier 3: High-Volume/Remote Operations (12+ months)

**Primary Method: Remote-Controlled Mini Excavator**

For surveys requiring remote operation from safe standoff distance.

| Aspect | Specification |
|--------|---------------|
| Equipment | 1-2 ton mini excavator + remote kit + custom attachment |
| Cost | $60,000-95,000 |
| Standoff | 100-300m remote operation |
| Crew | 1 operator + 1 spotter |
| Production | 15-30 holes/day |

**Configuration:**
- Used mini excavator (1-2 ton class)
- Stanley ROC remote control kit
- Custom probe insertion attachment
- Probe magazine (10-15 probes)

**Why This for High-Risk Sites:**
- Removes personnel from UXO hazard zone
- Excavator weight provides reaction force
- Proven remote control technology
- Can also prepare site access

---

## Method-by-Method Summary

### 1. Hydraulic Static Push (CPT-Style)

**Assessment:** HIGHLY FEASIBLE - Recommended as primary production method

**Key Findings:**
- Standard CPT equipment is vastly overpowered for HIRT (10-20x margin)
- Push force requirement: 5-15 kN for 16mm probes (vs 100-200 kN CPT capacity)
- Industry already uses CPT+magnetometer for UXO site investigation
- Production rate: 20-50 holes/day with experienced crew

**Equipment Options:**

| Option | Push Force | Cost | Notes |
|--------|------------|------|-------|
| Gouda 200kN Stand-Alone | 200 kN | $50-80K | Proven, anchor-based |
| Gouda 100kN Stand-Alone | 100 kN | $40-60K | Adequate for HIRT |
| Mini CPT Crawler | 100-200 kN | $100-200K | Self-propelled |
| Purpose-built HIRT rig | 10-50 kN | $25-40K | Optimized for application |

**Recommendation:** Develop purpose-built lightweight rig (~$10K) first; consider Gouda 100kN for commercial operations.

**References:** [Full assessment document](feasibility-hydraulic-push.md)

---

### 2. Water Jet Systems

**Assessment:** HIGHLY FEASIBLE - Recommended as Tier 1 method

**Key Findings:**
- Water jetting achieves 2-3m depth in sandy/silty soils easily
- Pressure requirement: 5-8 bar (standard high-pressure pump)
- Water consumption: ~100L per hole (50L with recirculation)
- UXO Safety: SAFE - hydraulic erosion, no mechanical impact

**System Configurations:**

| Configuration | Cost | Complexity | Best For |
|---------------|------|------------|----------|
| Pressure washer + lance | $500-700 | Low | Testing, light use |
| Dedicated pump system | $1,400-2,000 | Medium | Standard operations |
| Semi-automated | $3,500-5,000 | Higher | Production surveys |

**Key Technique:** Hybrid jetting + push
1. Water jet creates 20mm pilot hole to full depth
2. Fill with bentonite slurry for stability
3. Insert 16mm probe into pre-formed hole
4. Minimal force required (slurry-lubricated)

**Limitations:**
- Gravel layers cause refusal
- Requires water supply (1,000-2,000L per 20-hole survey)
- Generates slurry requiring management

**References:** [Full assessment document](feasibility-water-jet.md)

---

### 3. Robotic Deployment

**Assessment:** TECHNICALLY FEASIBLE - Recommended for high-risk UXO sites

**Key Findings:**
- Removes personnel from UXO hazard zone (100-300m standoff)
- Force requirements (5-20 kN) achievable with proper anchoring
- Near-term: Remote-controlled mini excavator ($60-95K)
- Long-term: Purpose-built robot HAPIR ($150-255K)

**Development Path:**

| Phase | Timeline | Cost | Capability |
|-------|----------|------|------------|
| Remote mini excavator | 3-6 months | $60-95K | Basic remote insertion |
| Enhanced positioning | 6-12 months | +$50-100K | RTK-GPS, semi-autonomous |
| Purpose-built robot | 12-24 months | +$100-200K | Full autonomous operation |

**Near-Term Solution:**
- Used 1-2 ton mini excavator: $15-30K
- Stanley ROC remote kit: $15-25K
- Custom insertion attachment: $10-20K
- Integration: $10-15K
- **Total: $50-90K**

**Safety Considerations:**
- Minimum standoff: 100m (reduced blast radius)
- Conservative standoff: 300m (German evacuation protocols)
- Emergency stop at multiple levels
- Robot is expendable (never risk personnel for recovery)

**References:** [Full assessment document](feasibility-robotic-deployment.md)

---

### 4. Biomimetic Root Growth

**Assessment:** NOT VIABLE for direct use - Root-inspired features recommended

**Key Findings:**
- Technology at TRL 3-4 (laboratory validation only)
- Speed too slow: 1-60 mm/min (50 minutes to 50+ hours per 3m hole)
- Depth demonstrated: only 250mm (HIRT needs 2,500-3,000mm)
- Material deposition incompatible with archaeology

**Viable Root-Inspired Adaptations:**

| Adaptation | Benefit | Implementation | Cost |
|------------|---------|----------------|------|
| Tip geometry optimization | 10-20% force reduction | CAD redesign | Minimal |
| Hydrophobic nano-coating | 30-50% friction reduction | Commercial service | $10-20/probe |
| Sliding sleeve system | 50-70% friction reduction | Mechanical development | $5-15K dev |
| Water-jet tip | 40-60% force reduction | Already in methods | Included |

**Recommendation:** Implement root-inspired passive features (tip geometry, coatings) on current probes. Monitor IIT research for long-term potential.

**References:** [Full assessment document](feasibility-biomimetic-root-growth.md)

---

### 5. Lightweight Purpose-Built Rig

**Assessment:** HIGHLY FEASIBLE - Best cost/capability ratio

**Concept Design: HMIS (HIRT Modular Insertion System)**

| Specification | Value |
|---------------|-------|
| Push force | 8-10 kN max |
| Weight | 75-100 kg total |
| Stroke | 1.5m (single stroke) |
| Power | Battery (48V, 10Ah) or 12V vehicle |
| Anchoring | 4x helical ground anchors |
| Control | 50m tethered remote |
| Transport | 2-person carry, fits in SUV |

**Cost Breakdown:**

| Component | Estimated Cost |
|-----------|----------------|
| Aluminum frame | $1,500-2,500 |
| Hydraulic cylinder (10kN, 1.5m) | $800-1,500 |
| Battery hydraulic pump | $1,500-2,500 |
| Ground anchors (4x) | $400-800 |
| Control electronics | $500-1,000 |
| Tethered control (50m) | $300-600 |
| Probe guide and holder | $200-400 |
| Assembly and testing | $1,000-2,000 |
| **Total** | **$6,200-11,300** |

**Production estimate:** 25-40 holes/day with 2-person crew

---

## Comparison Matrix

### Method Comparison by Criteria

| Criterion | Water Jet | Hydraulic Push | Robot | Root Growth | Purpose Rig |
|-----------|-----------|----------------|-------|-------------|-------------|
| **Equipment Cost** | $500-5K | $50-200K | $60-255K | N/A | $6-12K |
| **Per-Hole Cost** | $3-10 | $50-90 | Variable | N/A | $20-40 |
| **UXO Safety** | SAFE | SAFE | SAFE | SAFE | SAFE |
| **Speed** | Medium | Fast | Medium | Very Slow | Fast |
| **Portability** | Excellent | Poor-Good | Poor | N/A | Excellent |
| **Soil Versatility** | Good | Excellent | Excellent | Limited | Good |
| **Remote Capable** | Yes | With mods | Yes | N/A | Yes |
| **Technical Risk** | Low | Low | Medium | High | Low |
| **Development Time** | 1-3 months | 3-6 months | 6-18 months | 8-15 years | 3-6 months |

### Recommended Method by Scenario

| Scenario | Primary Method | Backup Method |
|----------|---------------|---------------|
| **Initial testing/development** | Water jet | Hand auger |
| **Low-risk sites** | Purpose-built rig | Water jet |
| **Standard operations** | Purpose-built rig | Hydraulic push |
| **High-risk UXO sites** | Remote excavator | Tethered push rig |
| **High-volume surveys** | Commercial CPT | Purpose-built rig |
| **Difficult access** | Water jet | Purpose-built rig |

---

## Implementation Roadmap

### Phase 1: Proof of Concept (Months 1-3)

**Objectives:**
- Validate water-jet technique for HIRT probes
- Test in representative soil conditions
- Establish baseline production rates

**Actions:**
1. Build basic water-jet system ($500-700)
2. Fabricate jetting lance with probe guide
3. Test in sandbox and field conditions
4. Measure water consumption, penetration rates
5. Document procedures

**Deliverables:**
- Validated water-jet procedure
- Equipment specifications
- Training materials

**Budget:** $2,000-3,000

### Phase 2: Optimized Manual System (Months 3-6)

**Objectives:**
- Develop production-ready water-jet system
- Design lightweight hydraulic rig

**Actions:**
1. Procure dedicated pump (Honda WH20)
2. Implement bentonite slurry system
3. Build recirculation system
4. Begin HMIS frame design
5. Field test at representative sites

**Deliverables:**
- Production water-jet system
- HMIS design specifications
- Field operation procedures

**Budget:** $5,000-8,000

### Phase 3: Lightweight Rig Development (Months 6-12)

**Objectives:**
- Build and test HMIS prototype
- Validate tethered remote control

**Actions:**
1. Fabricate HMIS frame and hydraulics
2. Integrate ground anchors
3. Develop 50m tethered control
4. Field testing and iteration
5. Operator training

**Deliverables:**
- Working HMIS prototype
- Operations manual
- Trained operators

**Budget:** $8,000-15,000

### Phase 4: Remote Operations (Months 12-24)

**Objectives:**
- Develop remote capability for high-risk sites
- Evaluate robotic options

**Actions:**
1. Procure used mini excavator
2. Install remote control kit
3. Develop custom insertion attachment
4. Field validation at UXO training site
5. Document safety procedures

**Deliverables:**
- Remote-controlled insertion system
- UXO site operation procedures
- Safety protocols

**Budget:** $60,000-95,000 (if proceeding)

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Gravel refusal (jetting) | Medium | Medium | Pre-survey auger probe, relocate holes |
| Clay clogging (jetting) | Medium | Low | Bentonite additives, pressure increase |
| Insufficient push force | Low | High | Design margin (10kN), pilot hole option |
| Anchor pullout | Low | Medium | Use 4 anchors, verify soil conditions |
| Equipment failure in field | Low | Medium | Spare parts, field repair kit |

### UXO-Specific Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Probe triggers UXO | Very Low | Critical | Static push only, no percussion |
| Personnel in blast zone | Variable | Critical | Tethered control, standoff distance |
| Robot stuck/abandoned | Low | Low | Design for abandonment, no hazmat |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Component delays | Medium | Low | Use off-shelf parts where possible |
| Weather delays (field tests) | Medium | Low | Schedule margin |
| Design iteration needed | High | Medium | Iterative development, budget contingency |

---

## Cost Summary

### Development Budget

| Phase | Budget | Cumulative |
|-------|--------|------------|
| Phase 1 (Proof of Concept) | $2,000-3,000 | $2,000-3,000 |
| Phase 2 (Optimized Manual) | $5,000-8,000 | $7,000-11,000 |
| Phase 3 (Lightweight Rig) | $8,000-15,000 | $15,000-26,000 |
| Phase 4 (Remote Ops) | $60,000-95,000 | $75,000-121,000 |

**Note:** Phase 4 is optional, depending on UXO site requirements.

### Per-Survey Operational Costs (After Development)

| Method | 20-Hole Survey | 50-Hole Survey |
|--------|----------------|----------------|
| Water jet | $200-400 | $400-750 |
| Purpose-built rig | $400-800 | $800-1,500 |
| Remote excavator | $1,500-3,000 | $3,000-5,000 |
| Commercial CPT contract | $4,000-8,000 | $10,000-15,000 |

---

## Conclusions

### Key Recommendations

1. **Start with water jetting** - Lowest cost, proven technique, immediate availability

2. **Develop lightweight purpose-built rig (HMIS)** - Best cost/capability ratio for standard operations

3. **Add remote capability for UXO sites** - Remote-controlled excavator provides personnel safety

4. **Adopt root-inspired features** - Passive improvements (tip geometry, coatings) reduce insertion force

5. **Monitor biomimetic technology** - Root-growth robots may become viable in 10+ years

### Final Assessment

The HIRT probe insertion challenge is well-served by existing technology, properly adapted:

- **Water jetting** provides an immediately deployable, low-cost solution
- **Hydraulic static push** is industry-standard for UXO sites
- **Purpose-built lightweight rigs** offer the best operational efficiency
- **Remote operation** is achievable for high-risk sites

No exotic or unproven technology is required. The key is proper engineering adaptation of proven methods to HIRT's specific requirements.

---

## Source Documents

### Feasibility Assessments Created

1. [Hydraulic Static Push Assessment](feasibility-hydraulic-push.md)
2. [Water Jet Drilling Assessment](feasibility-water-jet.md)
3. [Robotic Deployment Assessment](feasibility-robotic-deployment.md)
4. [Biomimetic Root Growth Assessment](feasibility-biomimetic-root-growth.md)

### Research Catalog

- [Borehole Creation Methods Catalog](borehole-creation-methods-catalog.md) - Comprehensive listing of 30+ methods

### External References

See individual assessment documents for detailed source citations.

---

*Document compiled: 2026-01-19*
*For HIRT Geophysical Survey System development*
