# Flight Difficulty Case Study — Presentation

Slide 1 — Title
- Flight Difficulty Scoring & Operational Insights
- Author: chaitanyakumar
- Date: (run date)

Slide 2 — Objective & Data
- Objective: build a daily, per-flight difficulty score to prioritize operational focus.
- Data: Flight-level, PNR (passenger), PNR remarks (SSRs), Bag-level, Airports reference.

Slide 3 — Approach
- EDA: compute delays, ground time surplus, bag aggregates, load factors, SSR counts.
- Feature engineering: load_factor, transfer_share, ground_pressure (negative ground_time_surplus), n_ssr, bag counts, passenger composition.
- Scoring: robust scaling (median/MAD) per-day, weighted linear combination, normalized 0..1, daily ranking and 3-way class.

Slide 4 — Key EDA results
- Average departure delay: 21.18 minutes
- % flights delayed: 49.6%
- Flights with scheduled ground time <= minimum_turn + 5min: 780
- Weak negative correlation between load and delays (corr=-0.150)

Slide 5 — Scoring method
- Features standardized within-day using median/MAD to reduce outlier effects.
- Weighted combination (example weights): ground_pressure 30%, load 20%, transfer_share 15%, n_ssr 15%, other small weights.
- Normalize to 0..1 each day; rank and classify: top 20% -> Difficult, bottom 30% -> Easy, rest -> Medium.

Slide 6 — Top operational findings
- Destinations with most Difficult flights: IAH, SFO, DEN, LAX, SEA (sample top 5)
- Common drivers: short ground_time_surplus, high transfer share, higher SSR counts in some flights, and aircraft/turn patterns.

Slide 7 — Recommendations
- Increase buffer for flights with ground_time_surplus <= minimum_turn + 5min, or prioritize resources (staffing/bridging) for high-rank flights.
- Review transfer baggage handling at frequent difficult destinations, add dedicated transfer lanes or pre-boarding sorting.
- Monitor SSR-heavy flights and pre-assign assistance staff.
- Use the daily score to prioritize real-time decisions: dispatch resources, pre-clear transfer bags, adjust pushback targets.

Slide 8 — Next steps & reproducibility
- Validate with more historical days and compare to actual delay incidents.
- Add model-based scoring (e.g., XGBoost) to predict minute-level delay risk and compare to rule-based score.
- Code: `hello.py` (main), `requirements.txt`. Run instructions in `DELIVERABLES.md`.
