# Dataset Improvement Plan

## Current State

| Metric | Value |
|---|---|
| Total .npy files | 1,332 |
| Unique words | 1,263 |
| Completely zero files | ~71 (5.3%) |
| Left-hand all-zero | ~962 (72%) |
| Have left-hand data | ~370 (28%) |
| Frames per sign | 1 (static pose) |
| Source images (.jpg) | 0 (all missing) |
| Frame_Count in CSV | 0 for all rows |

## Issues

1. **No temporal motion** — Every .npy is a single (258,) static frame. The pipeline interpolates to 30 frames, but since input is one frame, output is the same pose repeated 30×. No actual movement is captured or rendered.

2. **Left hand always zero** — 72% of files have zero left-hand landmarks (indices 132-195). MediaPipe Holistic outputs left-hand data when the left hand is visible, but the collection pipeline either didn't run MediaPipe on full-body video or post-processed incorrectly.

3. **~5% completely dead files** — 71 files are all zeros (e.g. the word for "donkey" / حمار). These provide no useful data.

4. **Source frames deleted** — The `Raw_Image_Path` column references .jpg files that no longer exist. Can't re-extract keypoints without re-recording.

## Recommended Collection Pipeline

Replace the current single-frame capture with a video-based MediaPipe Holistic pipeline:

```
Video recording (3-5 sec per sign)
        ↓
MediaPipe Holistic (full frame, both hands)
        ↓
Extract pose (33×4), left hand (21×3), right hand (21×3)
        ↓
Concatenate to (n_frames, 258) array  ← temporal sequence
        ↓
Save as .npy with frame_count metadata
        ↓
Update CSV: Word, Frame_Count=n, Keypoints_Path, Raw_Image_Path
```

### Requirements

- Record at 30 fps, ~3-5 seconds per sign (90-150 frames)
- Ensure both hands are fully visible and well-lit
- Use MediaPipe Holistic (not Pose) for left-hand landmarks
- Validate each recording: skip if hand visibility < 70% of frames

### Validation Checklist

Each new .npy file should pass:

- [ ] Shape is (n, 258) with n >= 30
- [ ] Not all-zero (L2 norm > 0.1)
- [ ] Left hand has non-zero data in at least 50% of frames
- [ ] Right hand has non-zero data in at least 50% of frames
- [ ] No NaN or Inf values

### Fixing Left-Hand Gap

The left hand all-zero issue is likely caused by:
1. Using MediaPipe Pose instead of MediaPipe Holistic (Pose doesn't track hands)
2. Recording with one hand (right) visible only
3. Post-processing bug that discarded left-hand data

**Fix**: Switch to `mediapipe.solutions.holistic.Holistic()`, which outputs all 543 landmarks including both hands in a single pass.

### Zero-Frame Cleanup

Remove these ~71 files from the manifest and re-record:
- They provide no useful data
- Current code now gracefully skips them (via `np.all(np.abs(arr) < 1e-6)` check)
- Flagged for re-recording

### Collection Scale

- Current: 1,263 unique words, 1,332 files (mostly 1:1)
- Target: At least 2-3 recordings per word for variation
- Priority: Re-record the ~71 zero files first, then bulk-record the missing left-hand signs
- Tools: OpenCV VideoCapture + MediaPipe Holistic, guided by existing word list

### Minimal First Step

If re-recording 1,263 signs is too large, start with:
1. Record 50 high-frequency words with full MediaPipe Holistic (both hands)
2. Validate manually
3. Scale up from there

### Integration

Updated `.npy` files go into `Dataset/Labels/{word}/keypoints/` as `frame_0.npy`, `frame_1.npy`, ... (each a full sequence).
The pipeline's `get_keypoints_for_word` already handles multi-frame stacking and interpolation, so no code changes needed.
