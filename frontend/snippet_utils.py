#snippet_utils.py
import os
import pandas as pd
import numpy as np
import tempfile
import uuid
import cv2
import moviepy.editor as mp

from detectron2_utils import process_frame


def merge_overlapping_boxes(boxes, scores, classes, iou_threshold):
    keep = []
    merged_boxes, merged_scores, merged_classes = [], [], []
    indices = list(scores.argsort()[::-1])
    while indices:
        idx = indices.pop(0)
        b, c, s = boxes[idx], classes[idx], scores[idx]
        x1, y1, x2, y2 = b
        new_indices = []
        for j in indices:
            if classes[j] != c:
                new_indices.append(j)
                continue
            b2 = boxes[j]
            # compute IoU
            ix1, iy1, ix2, iy2 = max(x1, b2[0]), max(y1, b2[1]), min(x2, b2[2]), min(y2, b2[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area1 = (x2-x1) * (y2-y1)
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            union = area1 + area2 - inter
            if union > 0 and inter / union >= iou_threshold:
                # merge coords
                x1, y1 = min(x1, b2[0]), min(y1, b2[1])
                x2, y2 = max(x2, b2[2]), max(y2, b2[3])
            else:
                new_indices.append(j)
        merged_boxes.append([x1, y1, x2, y2])
        merged_scores.append(s)
        merged_classes.append(c)
        indices = new_indices

    return (np.array(merged_boxes), np.array(merged_scores), np.array(merged_classes))


def select_bboxes(bboxes, indices):
    return [bboxes[i] for i in indices if i < len(bboxes)]

def bbox_overlap(box1, box2):
    # Standard bbox overlap for [x1, y1, x2, y2]
    return (box1[0] < box2[2] and box1[2] > box2[0]) and \
           (box1[1] < box2[3] and box1[3] > box2[1])

def is_significantly_different(box1, box2, threshold=0.15):
    if not box1 or not box2:
        return False
    dx1 = abs(box1[0] - box2[0])
    dy1 = abs(box1[1] - box2[1])
    dx2 = abs(box1[2] - box2[2])
    dy2 = abs(box1[3] - box2[3])
    avg_width = ((box1[2] - box1[0]) + (box2[2] - box2[0])) / 2
    avg_height = ((box1[3] - box1[1]) + (box2[3] - box2[1])) / 2
    return (dx1 / avg_width >= threshold) or (dy1 / avg_height >= threshold) or \
           (dx2 / avg_width >= threshold) or (dy2 / avg_height >= threshold)

def start_stop_sniffing(
    bounding_box_location_tank_head,
    threshold_bb=0.15,
    fps=30,
    vid_nr="snippet"
):
    bb = pd.DataFrame(bounding_box_location_tank_head)
    bb['indexes_nose'] = bb["pred_classes"].apply(lambda x: [i for i, v in enumerate(x) if v == 0])
    bb['indexes_tank'] = bb["pred_classes"].apply(lambda x: [i for i, v in enumerate(x) if v == 1])
    bb['bboxes_tanks'] = bb.apply(lambda row: select_bboxes(row['bounding_box'], row['indexes_tank']), axis=1)
    bb['bboxes_nose'] = bb.apply(lambda row: select_bboxes(row['bounding_box'], row['indexes_nose']), axis=1)
    bb['sniffing'] = bb.apply(
        lambda row: (
            'sniffing' if any(
                bbox_overlap(n, t)
                for n in row['bboxes_nose']
                for t in row['bboxes_tanks']
            ) else 'not_sniffing'
        ),
        axis=1
    )
    bb['prev_sniffing'] = bb['sniffing'].shift(1)
    bb['next_sniffing'] = bb['sniffing'].shift(-1)
    bb['prev_bboxes_nose'] = bb['bboxes_nose'].shift(1)
    bb['prev_bboxes_tanks'] = bb['bboxes_tanks'].shift(1)
    bb['start-stop'] = "continue"
    bb['bboxes_nose'] = bb['bboxes_nose'].apply(lambda x: x if isinstance(x, list) else [])
    bb['prev_bboxes_nose'] = bb['prev_bboxes_nose'].apply(lambda x: x if isinstance(x, list) else [])
    bb.loc[bb['prev_sniffing'].isna(), 'start-stop'] = 'start'
    bb.loc[
        (bb['prev_sniffing'] != 'sniffing') &
        (bb['sniffing'] == 'sniffing') &
        (bb.apply(lambda row: any(is_significantly_different(n, pn, threshold=threshold_bb)
                                  for n in row['bboxes_nose'] for pn in row['prev_bboxes_nose']), axis=1)),
        'start-stop'
    ] = 'start'
    bb.loc[
        (bb['sniffing'] != 'sniffing') &
        (bb['prev_sniffing'] == 'sniffing') &
        (bb.apply(lambda row: any(is_significantly_different(n, pn, threshold=threshold_bb)
                                  for n in row['bboxes_nose'] for pn in row['prev_bboxes_nose']), axis=1)),
        'start-stop'
    ] = 'stop'
    bb.loc[
        (bb['next_sniffing'].isna()) &
        (bb['sniffing'] == 'sniffing') &
        (bb.apply(lambda row: any(is_significantly_different(n, pn, threshold=threshold_bb)
                                  for n in row['bboxes_nose'] for pn in row['prev_bboxes_nose']), axis=1)),
        'start-stop'
    ] = 'stop'
    bb1 = bb.copy()
    bb1 = bb1[bb1['start-stop'].isin(['start', 'stop', 'stop2'])]
    bb2 = bb1.copy()
    bb2.drop(columns=['prev_sniffing', 'next_sniffing'], inplace=True)
    bb3 = bb2.copy()
    bb3 = bb3[bb3['start-stop'] == 'start']
    bb4 = bb2.copy()
    bb4 = bb2[bb2['start-stop'] == 'stop']
    bb3.reset_index(drop=True, inplace=True)
    bb4.reset_index(drop=True, inplace=True)
    bb3['stop_timestamp'] = bb4['timestamp']
    bb3.drop(columns=['sniffing', 'start-stop'], inplace=True)
    bb4 = bb3.copy()
    bb4.rename(columns={'timestamp': 'start_timestamp'}, inplace=True)
    frame_count = bb4['frame_number'].max() 
    last_timestamp = frame_count / fps if not pd.isna(frame_count) else 0
    if pd.isna(bb4.loc[bb4.index[-1], 'stop_timestamp']):
        bb4.loc[bb4.index[-1], 'stop_timestamp'] = last_timestamp
    bb4['start_timestamp'] = bb4['start_timestamp'].astype(float)
    bb4['stop_timestamp'] = bb4['stop_timestamp'].astype(float)
    bb4['new_stop_timestamp'] = bb4['start_timestamp'].shift(-1).astype(float)
    bb4['duration'] = bb4['stop_timestamp'] - bb4['start_timestamp']
    bb4['new_duration'] = bb4['new_stop_timestamp'] - bb4['start_timestamp']
    last_timestamp = bb4['start_timestamp'].max() if not bb4['start_timestamp'].empty else 0
    if pd.isna(bb4.loc[bb4.index[-1], 'stop_timestamp']):
        bb4.loc[bb4.index[-1], 'stop_timestamp'] = last_timestamp
    bb4['new_stop_timestamp'] = bb4['new_stop_timestamp'].fillna(bb4['stop_timestamp'])
    bb4['stop_timestamp'] = bb4['stop_timestamp'].astype(float)
    bb4['start_timestamp'] = bb4['start_timestamp'].astype(float)
    bb4['new_stop_timestamp'] = bb4['new_stop_timestamp'].astype(float)
    bb4['duration'] = bb4['stop_timestamp'] - bb4['start_timestamp']
    bb4['new_duration'] = bb4['new_stop_timestamp'] - bb4['start_timestamp']
    bb4['snippet_name'] = vid_nr + "_snippet_" + bb4.index.astype(str) + ".mp4"
    bb4["video_filename"] = vid_nr + "processed_output.mp4"
    return bb4


def generate_snippets(
    input_video_path,
    predictor,
    conf_thresh=0.3,
    threshold_bb=0.15,
    iou_threshold=0.00001
):
    """
    1) Run predictor on the original video, build bb_list
    2) Compute sniff start/stop with start_stop_sniffing()
    3) Cut each snippet from ORIGINAL, name them {basename}_snippet_{idx}.mp4
    4) Return list of snippet file paths
    """
    # 1) gather predictions per frame
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_base = os.path.splitext(os.path.basename(input_video_path))[0]

    bb_list = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        inst = predictor(frame)["instances"].to("cpu")
        boxes   = inst.pred_boxes.tensor.numpy()
        scores  = inst.scores.numpy()
        classes = inst.pred_classes.numpy()
        if boxes.size > 0:
            boxes, scores, classes = merge_overlapping_boxes(
                boxes, scores, classes, iou_threshold
            )
        keep = scores >= conf_thresh
        bb_list.append({
            "frame_number": i,
            "timestamp": i / fps,
            "bounding_box": [b.tolist() for b in boxes[keep]],
            "pred_classes": classes[keep].tolist()
        })
    cap.release()

    # 2) detect sniff start/stop
    df = start_stop_sniffing(
        bb_list,
        threshold_bb=threshold_bb,
        fps=fps,
        vid_nr=vid_base
    )
    if df.empty:
        return []

    # 3) prepare output directory
    out_dir = tempfile.mkdtemp(prefix=f"{vid_base}_snips_")
    snippet_paths = []

    # 4) cut each snippet
    for idx, row in df.reset_index().iterrows():
        start_t = row["start_timestamp"]
        stop_t  = row["new_stop_timestamp"]
        out_path = os.path.join(out_dir, f"{vid_base}_snippet_{idx}.mp4")

        try:
            ( mp.VideoFileClip(input_video_path)
               .subclip(start_t, stop_t)
               .without_audio()
               .write_videofile(
                   out_path,
                   codec="libx264",
                   audio=False,
                   verbose=False,
                   logger=None
               )
            )
            snippet_paths.append(out_path)
        except Exception as e:
            print(f"[generate_snippets] error writing {out_path}: {e}")

    return snippet_paths

    