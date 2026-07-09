# GameVision AI Evaluation Summary

## Evaluation Methodology

GameVision AI was evaluated across three layers:

1. **Object detection** using a held-out validation split.
2. **Runtime performance** on the sample football match clip.
3. **Tracking behavior** using descriptive continuity and temporal coverage statistics derived from predicted ByteTrack trajectories.

Tracking statistics are descriptive metrics computed from predicted tracks. They are not ground-truth MOT metrics such as MOTA, HOTA, or IDF1.

---

## Overall Detection Performance

| Metric | Value |
|---|---:|
| Precision | 0.9195 |
| Recall | 0.8300 |
| mAP@0.5 | 0.8601 |
| mAP@0.5:0.95 | 0.5831 |

---

## Class-wise Detection Performance

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Player | 0.9712 | 0.9877 | 0.9919 | 0.7469 |
| Referee | 0.9060 | 0.9612 | 0.9625 | 0.7213 |
| Ball | 0.8814 | 0.5410 | 0.6259 | 0.2811 |

---

## Runtime Benchmark

| Metric | Value |
|---|---:|
| Effective throughput | 6.54 FPS |
| Average latency | 152.95 ms/frame |
| Frames processed | 151 |
| Source FPS | 25.00 |

The runtime benchmark measures detector inference performance in the local execution environment. Runtime results are hardware-dependent and should not be interpreted as universal deployment performance.

---

## Track Temporal Coverage

| Class | Unique Track IDs | Mean Coverage | Median Coverage | Mean Missing Frames | Longest Observed Track |
|---|---:|---:|---:|---:|---:|
| Player | 39 | 0.841 | 0.985 | 8.28 | 151 |
| Referee | 12 | 0.700 | 0.677 | 21.83 | 145 |
| Ball | 7 | 0.941 | 1.000 | 2.43 | 30 |

---

## Interpretation

The detector performs strongly on the player and referee classes, while ball detection remains the primary small-object detection challenge.

Player trajectories demonstrate strong temporal continuity across the evaluated clip. Referee trajectories exhibit greater fragmentation and more missing observations within track spans.

Ball tracks show high within-track temporal coverage but are distributed across multiple track IDs. This indicates that once the ball is successfully associated with a track, short-term continuity can be strong, while detection gaps and rapid motion can still cause identity fragmentation.

---

## Engineering Observations

- Player detection is the strongest component of the detector, with high precision, recall, and mAP.
- Referee detection is also strong on the held-out validation set, although tracking continuity is weaker than for players.
- Ball detection has substantially lower recall than player and referee detection, reflecting the difficulty of detecting small, fast-moving objects in compressed broadcast footage.
- ByteTrack provides persistent identities across frames, enabling trajectory generation and per-track motion analysis.
- Temporal coverage metrics provide descriptive insight into predicted track continuity without claiming unsupported ground-truth MOT accuracy.

---

## Limitations

- Tracking statistics are derived from predicted trajectories rather than ground-truth MOT annotations.
- Therefore, MOTA, HOTA, and IDF1 are not reported.
- Runtime measurements are hardware-dependent.
- Ball detection performance is constrained by small object size, motion blur, occlusion, and broadcast-video compression.
- Image-coordinate movement statistics are affected by camera motion and perspective.
- Pixel displacement and pixel speed should not be interpreted as real-world physical distance or velocity.
- Evaluation results are specific to the current held-out validation split and sample match footage.

---

## Generated Artifacts

This evaluation pipeline generates:

- Detection metrics in JSON format
- Runtime benchmark results
- Tracking continuity statistics
- Track fragmentation and temporal coverage statistics
- Consolidated JSON evaluation report
- Consolidated Markdown evaluation report
- Player trajectory visualization
- Player occupancy heatmap
- Ball trajectory visualization

These artifacts provide both quantitative and qualitative evidence for evaluating the GameVision AI pipeline.
