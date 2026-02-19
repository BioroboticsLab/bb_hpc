# note!  no imports here! 
# all job functions need to be self-contained in order to work with the slurmhelper libraries, because with this, the functions are copied into running directories
# if a job function depends on any outside definitions or imports, it will fail with slurmhelper / hpc

#################################################################
##### DETECT (main pipeline) — self-contained for slurmhelper ####
#################################################################
def job_for_process_videos(
    *,
    video_paths=None,
    repo_output_path=None,
    timestamp_format="basler",
    num_threads=1,
    text_root_path=None,
    video_file_type="basler",
    # optional local staging to reduce shared-storage I/O:
    copy_local=False,
    local_cache_dir="/tmp/bb_localcache",
):
    """
    Run bb_pipeline.process_video on a list of video paths.

    - Self-contained (safe for slurmhelper).
    - If copy_local=True, each video (and its .txt sidecar if present) is copied
      to `local_cache_dir` before processing, then cleaned up afterwards.
    """
    if not video_paths:
        return True
    if repo_output_path is None:
        raise ValueError("repo_output_path must be provided")

    # Local helpers kept inline so the function remains self-contained.
    import os, shutil, gc
    from pathlib import Path

    def _safe_makedirs(p: Path):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _copy_file(src: Path, dst: Path):
        _safe_makedirs(dst.parent)
        try:
            shutil.copy2(src, dst)
        except Exception:
            # very unlikely fallback
            shutil.copyfile(src, dst)
            try:
                shutil.copystat(src, dst)
            except Exception:
                pass

    def _stage_to_local(src_vid: Path) -> tuple[Path, Path | None]:
        """
        Copy src video and optional .txt sidecar to local_cache_dir.
        Returns (vid_local, txt_local_or_None). On failure, returns originals.
        """
        try:
            rel_parts = src_vid.parts[-3:] if len(src_vid.parts) >= 3 else src_vid.parts
            dst_vid = Path(local_cache_dir, *rel_parts)
            _copy_file(src_vid, dst_vid)

            txt = src_vid.with_suffix(".txt")
            dst_txt = None
            if txt.exists():
                dst_txt = dst_vid.with_suffix(".txt")
                _copy_file(txt, dst_txt)
            return dst_vid, dst_txt
        except Exception:
            return src_vid, (src_vid.with_suffix(".txt") if src_vid.with_suffix(".txt").exists() else None)

    def _cleanup_staged(paths: list[Path | None]):
        for p in paths:
            try:
                if p and p.exists():
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        # best-effort prune a couple levels
        try:
            for child in sorted(Path(local_cache_dir).glob("**/*"), reverse=True):
                if child.is_dir():
                    try:
                        next(child.iterdir())
                    except StopIteration:
                        child.rmdir()
        except Exception:
            pass

    # Build the simple args object expected by pipeline.scripts.bb_pipeline.process_video
    class _Args:
        pass

    args = _Args()
    args.num_threads      = num_threads
    args.timestamp_format = timestamp_format
    args.text_root_path   = text_root_path
    args.progressbar      = False
    args.video_path       = None
    args.video_file_type  = video_file_type
    args.repo_output_path = repo_output_path

    # Import inside the function (slurmhelper-safe)
    from pipeline.scripts.bb_pipeline import process_video

    _safe_makedirs(Path(local_cache_dir))

    for vp in video_paths:
        src = Path(vp)
        vid_for_proc = src
        staged_txt = None

        try:
            if copy_local:
                vid_for_proc, staged_txt = _stage_to_local(src)

            args.video_path = str(vid_for_proc)
            process_video(args)

        except Exception as e:
            print(f"[WARN] error processing {src}: {e}")
        finally:
            args.video_path = None
            gc.collect()
            # cleanup staged files to avoid filling the node
            if copy_local and vid_for_proc != src:
                _cleanup_staged([vid_for_proc, staged_txt])

    return True

#################################################################
##### SAVE DETECT 
#################################################################
def job_for_save_detect_chunk(job_args_list):
    """
    Process a list of one-hour detection segments in a single task.
    Each element in job_args_list is a dict of kwargs for save_all_detections.
    """

    def save_all_detections(repo_path, save_path, from_dt, to_dt, cam_id):
        """
        Extract detections from a bb_binary repository, convert them to the desired format,
        and save them to a Parquet file.
    
        Parameters:
        - repo_path: Path to the bb_binary repository.
        - save_path: Directory where the output Parquet file will be saved.
        - from_dt: Start datetime for extraction (datetime object or string in '%Y-%m-%d %H:%M:%S' format).
        - to_dt: End datetime for extraction (datetime object or string in '%Y-%m-%d %H:%M:%S' format).
        - cam_id: Camera ID to filter detections.
        """
    
        import pandas as pd
        import numpy as np
        import os
        import pytz
        from datetime import datetime
        from bb_tracking.data_walker import iterate_bb_binary_repository
        from bb_utils.ids import BeesbookID
        from enum import Enum
        from bb_binary.parsing import get_video_fname
        import capnp

    
        # Important variables
    
    
        # Detection Types (integer).  it was previously set as an 'enum', but this gave problems
        # class DetectionType(Enum):
        # TaggedBee = 'TaggedBee'  ## 1
        # UntaggedBee = 'UntaggedBee'  ## 2
        # BeeOnGlass = 'BeeOnGlass'  ## 3
        # BeeInCell = 'BeeInCell'   ## 4  
    
        # Define subfunction to get detections from bb_binary
        def get_detections_from_bb_binary(repository_path, dt_begin, dt_end,
                                          only_tagged_bees=False, cam_id=None,
                                          is_2019_repos=True):
            detections_list = []
            # Use the iterator to get detections from the repository
            iterator = iterate_bb_binary_repository(
                repository_path, dt_begin, dt_end,
                only_tagged_bees=only_tagged_bees,
                cam_id=cam_id,
                is_2019_repos=is_2019_repos
            )
            while True:
                try:
                    cam_id_iter, frame_id, frame_datetime, frame_detections, frame_kdtree = next(iterator)
                except StopIteration:
                    break
                except ValueError as e:
                    # cKDTree construction failed for this frame—skip it
                    print(f"Skipping frame - (bad KD-tree): {e}")
                    continue     
                except capnp.KjException as e:
                    # Cap'n Proto Premature EOF — skip & log everything we know
                    # If you have a variable holding the .bb file path (e.g. fc_path), include that too:
                    print('capnp KJException')
                    # print(
                    #     f"[capnp EOF] frame={frame_id!r}, time={frame_datetime!r}, "
                    #     f"file={locals().get('fc_path','<unknown>')}: {type(e).__name__}: {e!r}"
                    # )
                    continue
            
                # (optional) catch any other unexpected errors, so your loop never dies
                except Exception as e:
                    print(
                        f"[unexpected error {type(e).__name__}] frame={frame_id!r}, "
                        f"time={frame_datetime!r}, file={locals().get('fc_path','<unknown>')}: {e!r}"
                    )
                    continue                    
            # for cam_id_iter, frame_id, frame_datetime, frame_detections, frame_kdtree in iterator:
                for det in frame_detections:
                    # Extract detection fields
                    detection_type = det.detection_type.value  # this will be an integer
                    x_pixels = det.x_pixels
                    y_pixels = det.y_pixels
                    orientation_pixels = det.orientation_pixels
                    timestamp = det.timestamp
                    detection_index = det.detection_index
                    # For tagged bees, get bee_id and confidence
                    if detection_type == 1:
                        # Convert bit probabilities to bits (0 or 1)
                        bits = (np.array(det.bit_probabilities) > 0.5).astype(int)
                        # Create a BeesbookID from bits
                        bb_id = BeesbookID.from_bb_binary(bits)
                        # Convert to integer ID
                        bee_id = bb_id.as_ferwar()
                        # Calculate confidence (product of probabilities)
                        bit_probs = np.array(det.bit_probabilities)   
                        # this confidence calculation is the same as bb_tracking -> track_generator -> calculate_tracked_bee_id
                        bee_id_confidence = np.prod(np.abs(bit_probs - 0.5) * 2.0)
                    else:
                        bee_id = None
                        bee_id_confidence = None
                    detections_list.append({
                        'timestamp':          det.timestamp,
                        'cam_id':             cam_id_iter,
                        'detection_type':     det.detection_type.value,
                        'x_pixels':           det.x_pixels,
                        'y_pixels':           det.y_pixels,
                        'orientation_pixels': det.orientation_pixels,
                        'localizer_saliency':  det.localizer_saliency, 
                        'bee_id':             bee_id,
                        'bee_id_confidence':  bee_id_confidence
                    })
            # Create a DataFrame from the list of detections
            return pd.DataFrame(detections_list)
    
        # Define subfunction to convert detections to df_untagged format
        def convert_detections_to_df_untagged(detections_df):
            """
    
            Parameters:
            - detections_df: pandas DataFrame with detection data
    
            Returns:
            - df_untagged: pandas DataFrame in the required format
            """
        
            # Ensure that data types are consistent
            columns_to_numeric = ['cam_id', 'detection_type', 'x_pixels', 'y_pixels', 'orientation_pixels', 'localizer_saliency', 'bee_id', 'bee_id_confidence']        
            
            for col in columns_to_numeric:
                detections_df[col] = pd.to_numeric(detections_df[col], errors='coerce')
    
            return detections_df
    
        # Parse from_dt and to_dt to datetime objects if they are strings
        if isinstance(from_dt, str):
            from_dt = datetime.strptime(from_dt, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC)
        else:
            from_dt = from_dt.astimezone(pytz.UTC)
        if isinstance(to_dt, str):
            to_dt = datetime.strptime(to_dt, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC)
        else:
            to_dt = to_dt.astimezone(pytz.UTC)
    
        # Get detections from bb_binary
        detections_df = get_detections_from_bb_binary(
            repository_path=repo_path,
            dt_begin=from_dt,
            dt_end=to_dt,
            only_tagged_bees=False,
            cam_id=cam_id,
            is_2019_repos=True
        )
    
        if len(detections_df)>0:
            # Convert detections to df_untagged format
            df_untagged = convert_detections_to_df_untagged(detections_df)
        else:
            df_untagged = pd.DataFrame()
    
        # Generate the output filename using the same method as bb_binary
        output_filename = os.path.join(save_path, get_video_fname(cam_id, from_dt, to_dt) + '.parquet')
    
        # Save to Parquet file
        df_untagged.to_parquet(output_filename)
    
    for kwargs in job_args_list:
        save_all_detections(**kwargs)
    return True

#################################################################
##### TRACKING
#################################################################
def job_for_tracking(repo_path, save_path, temp_path, from_dt, to_dt, cam_id):
    import bb_tracking.models, bb_tracking.features
    import bb_tracking.repository_tracker
    import os
    import dill
    import shutil

    # use the same file-name generating function used by bb_binary
    from bb_binary.parsing import get_video_fname

    # ---- Compatibility patch for legacy repos with duplicate segments (no symlinks) ----
    # For newer repos this is a no-op (timestamps are already monotonic).
    try:
        import bb_tracking.data_walker as _dw
        _orig_iterate = getattr(_dw, "iterate_bb_binary_repository", None)
    except Exception:
        _orig_iterate = None

    if _orig_iterate is not None:
        from pathlib import Path
        import numpy as _np
        import datetime as _dt
        import pytz as _pytz

        def _iter_wrap(repository_path, dt_begin, dt_end, homography_fn=None, **kwargs):
            """
            Wrap the original iterator:
            - De-dup frame containers by (cam_id, first_ts, last_ts)
            - Enforce per-camera strictly increasing frame timestamps (skip non-increasing)
            This safely handles legacy/tape repos where bucket-edge copies exist as real files.
            """
            # Call through to the original generator (note: homography_fn is positional)
            gen = _orig_iterate(repository_path, dt_begin, dt_end, homography_fn, **kwargs)

            last_ts_by_cam = {}
            seen_containers = set()

            def _emit_with_guard(cam_id, frames_buf):
                nonlocal last_ts_by_cam
                for (frame_id, frame_dt, frame_detections, frame_kdtree) in frames_buf:
                    ts = frame_dt.timestamp()
                    last = last_ts_by_cam.get(cam_id)
                    if (last is not None) and (ts <= last):
                        continue
                    last_ts_by_cam[cam_id] = ts
                    yield (cam_id, frame_id, frame_dt, frame_detections, frame_kdtree)

            try:
                buffered = []
                prev_cam = None
                prev_frame_id = None

                while True:
                    try:
                        cam_id_it, frame_id, frame_dt, frame_detections, frame_kdtree = next(gen)
                    except StopIteration:
                        if buffered:
                            first_ts = buffered[0][1].timestamp()
                            last_ts  = buffered[-1][1].timestamp()
                            key = (prev_cam, first_ts, last_ts)
                            if key not in seen_containers:
                                seen_containers.add(key)
                                for out in _emit_with_guard(prev_cam, buffered):
                                    yield out
                        break

                    new_container = (
                        prev_cam is None
                        or cam_id_it != prev_cam
                        or (prev_frame_id is not None and frame_id < prev_frame_id)
                    )

                    if new_container and buffered:
                        first_ts = buffered[0][1].timestamp()
                        last_ts  = buffered[-1][1].timestamp()
                        key = (prev_cam, first_ts, last_ts)
                        if key not in seen_containers:
                            seen_containers.add(key)
                            for out in _emit_with_guard(prev_cam, buffered):
                                yield out
                        buffered = []

                    buffered.append((frame_id, frame_dt, frame_detections, frame_kdtree))
                    prev_cam = cam_id_it
                    prev_frame_id = frame_id

            except Exception:
                # Fallback: pass-through with per-frame monotonic guard
                last_ts_by_cam = {}
                try:
                    gen2 = _orig_iterate(repository_path, dt_begin, dt_end, homography_fn, **kwargs)
                    for cam_id_it, frame_id, frame_dt, frame_detections, frame_kdtree in gen2:
                        ts = frame_dt.timestamp()
                        last = last_ts_by_cam.get(cam_id_it)
                        if (last is not None) and (ts <= last):
                            continue
                        last_ts_by_cam[cam_id_it] = ts
                        yield (cam_id_it, frame_id, frame_dt, frame_detections, frame_kdtree)
                except Exception:
                    # Last resort: yield original without changes
                    gen3 = _orig_iterate(repository_path, dt_begin, dt_end, homography_fn, **kwargs)
                    for item in gen3:
                        yield item

        # Monkey-patch
        _dw.iterate_bb_binary_repository = _iter_wrap
    # ---- end compatibility patch ----
    
    output_filename_tmp = os.path.join(temp_path,get_video_fname(cam_id, from_dt, to_dt)+'.dill.tmp')
    output_filename = os.path.join(save_path,get_video_fname(cam_id, from_dt, to_dt)+'.dill')

    # these thresholds are the same as David used:  0.6 and 0.5
    detection_classification_threshold = 0.6
    tracklet_classification_threshold = 0.5

    model_dir = os.getenv('CONDA_PREFIX') + '/pipeline_models'
    detection_model_path = os.path.join(model_dir, 'detection_model_4.json')
    tracklet_model_path = os.path.join(model_dir,'tracklet_model_8.json')
    detection_model = bb_tracking.models.load_xgb_model(detection_model_path)
    tracklet_model = bb_tracking.models.load_xgb_model(tracklet_model_path)

    px_per_cm_approx = 4833/43.2  # convert previously used thresholds to pixels, because now not applying the homography in the first step

    detection_kwargs = dict(
        max_distance_per_second = 30.0*px_per_cm_approx,
        n_features=18,
        detection_feature_fn=bb_tracking.features.get_detection_features,
        detection_cost_fn=lambda features: 1 - detection_model.predict_proba(features)[:, 1],
        max_cost=1.0 - detection_classification_threshold
        )

    tracklet_kwargs = dict(
        max_distance_per_second = 20.0*px_per_cm_approx,
        n_features=14,
        max_seconds_gap=4.0,
        tracklet_feature_fn=bb_tracking.features.get_track_features,
        tracklet_cost_fn=lambda features: 1 - tracklet_model.predict_proba(features)[:, 1],
        max_cost=1.0 - tracklet_classification_threshold
        )

    tracker = bb_tracking.repository_tracker.RepositoryTracker(repo_path,
                                                                    dt_begin=from_dt, dt_end=to_dt,
                                                                      cam_ids=(cam_id,),
                                                                      tracklet_kwargs=detection_kwargs,
                                                                      track_kwargs=tracklet_kwargs,
                                                                      repo_kwargs=dict(only_tagged_bees=True, fix_negative_timestamps=True),
                                                                      progress_bar=None, use_threading=False)
     
    # open the output file and write as results are generated
    def process_track(track, cam_id, results_file):
        batch = []
        for det in track.detections:
            batch.append((det.timestamp,
                          det.frame_id,
                          track.id,
                          det.x_pixels, det.y_pixels, det.orientation_pixels,
                          det.detection_index, det.detection_type.value,
                          cam_id, track.bee_id, track.bee_id_confidence))
        
        # Incrementally save the results batch into the .dill file
        dill.dump(batch, results_file)

    with open(output_filename_tmp, "wb") as results_file:
        # Get an iterator from the generator
        tracker_iter = iter(tracker)
        track = None 
        
        while True:
            try:
                # Try to get the next track from the generator
                track = next(tracker_iter)
            except StopIteration:
                print('reached the end')
                break
            except Exception as e:
                print(f"Error occurred while retrieving the next track: {e}")
                import traceback; traceback.print_exc()
                # inspect problematic detections (avoid relying on __dict__ on capnp objects)
                if hasattr(track, 'detections'):
                    for det in track.detections[:5]:
                        try:
                            # Print a compact summary
                            print("Det:",
                                  getattr(det, "timestamp", None),
                                  getattr(det, "frame_id", None),
                                  getattr(det, "detection_index", None),
                                  getattr(det, "x_pixels", None),
                                  getattr(det, "y_pixels", None))
                        except Exception:
                            # best-effort
                            print("Det:<unprintable>")
                continue 
            try:
                process_track(track, cam_id, results_file)
            except Exception as e:
                print(f"Error processing track {track.id}: {e}")
                if hasattr(track, 'timestamps'):
                    print(f"Timestamps: {track.timestamps}")
                continue  # Continue to the next track if processing failed
    # If successful, rename the file by removing the '.tmp' extension
    shutil.move(output_filename_tmp, output_filename)
    print(f"Processing completed: {output_filename}")    

#################################################################
##### RPI - DETECT
#################################################################
def job_for_process_rpi_videos(video_paths=None, clahe=True):
    import bb_behavior.tracking
    import pandas as pd
    from datetime import datetime
    import os

    def detect_rpi_video(videofile, n_frames=None, clahe=True):
        filename = os.path.basename(videofile)
        cam_id = filename.split('_')[0]
        timestamp_str = filename.split('_')[-1].replace('.h264', '')
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
        tag_pixel_diameter = 60.0
        confidence_filter = 0.001
        frame_info, video_dataframe = bb_behavior.tracking.detect_markers_in_video(
            videofile,
            timestamps=None,
            fps=10,
            cam_id=cam_id,
            clahe=clahe,     # <-- the knob
            n_frames=n_frames,
            tag_pixel_diameter=tag_pixel_diameter,
            confidence_filter=confidence_filter,
            use_parallel_jobs=False,
            progress=None
        )
        if video_dataframe is None:
            video_dataframe = pd.DataFrame()
        video_dataframe['video_start_timestamp'] = timestamp
        return video_dataframe

    suffix = "-c" if clahe else "-nc"

    for videofile in (video_paths or []):
        # idempotent
        det_path = os.path.splitext(videofile)[0] + f"-detections{suffix}.parquet"
        if os.path.exists(det_path):
            print(f"Skipping already processed video: {videofile}")
            continue

        df = detect_rpi_video(videofile, clahe=clahe)
        dirpath, basename = os.path.split(videofile)
        savename = basename.replace(".h264", f"-detections{suffix}.parquet")
        df.to_parquet(os.path.join(dirpath, savename))

    return True