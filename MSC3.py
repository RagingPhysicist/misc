import os
import numpy as np
import pandas as pd
import pydicom


def calculate_mcs_from_rtplan(rtplan_dicom_path):
    try:
        ds = pydicom.dcmread(rtplan_dicom_path)
    except Exception as e:
        print(f"Error reading DICOM file: {e}")
        return None, []

    if not hasattr(ds, "BeamSequence"):
        print("No BeamSequence found in RTPLAN.")
        return None, []

    beam_mcs_scores = []
    all_segment_details = []

    for beam in ds.BeamSequence:
        beam_type = beam.get("BeamType", "UNKNOWN").upper()
        if beam_type not in ("ARC", "DYNAMIC"):
            continue

        if not hasattr(beam, "ControlPointSequence"):
            continue

        cp_seq = beam.ControlPointSequence
        leaf_openings = []
        mu_weights = []

        for cp in cp_seq:
            if not hasattr(cp, "BeamLimitingDevicePositionSequence"):
                continue

            mlc_positions = [
                b for b in cp.BeamLimitingDevicePositionSequence
                if b.RTBeamLimitingDeviceType in ["MLCX", "MLCY"]
            ]
            if not mlc_positions:
                continue

            leaf_pos = np.array(mlc_positions[0].LeafJawPositions)
            n = len(leaf_pos) // 2
            opening = np.abs(leaf_pos[n:] - leaf_pos[:n])

            leaf_openings.append(opening)
            mu_weights.append(cp.CumulativeMetersetWeight)

        if len(leaf_openings) < 2:
            continue

        leaf_openings = np.array(leaf_openings)
        mu_weights = np.array(mu_weights)
        segment_weights = np.diff(mu_weights, prepend=0)

        non_zero_idx = np.where(~np.isclose(segment_weights, 0.0, atol=1e-5))[0]
        if len(non_zero_idx) == 0:
            continue

        segment_weights = segment_weights[non_zero_idx]
        leaf_openings = leaf_openings[non_zero_idx]

        total_weight = np.sum(segment_weights)
        if total_weight == 0:
            continue

        norm_weights = segment_weights / total_weight
        aperture_areas = np.array([np.sum(o) for o in leaf_openings])
        mean_area = np.mean(aperture_areas)
        if mean_area == 0:
            continue

        aav_scores = aperture_areas / mean_area
        beam_segment_details = []

        for i, opening in enumerate(leaf_openings):
            diffs = np.abs(np.diff(opening))
            max_vals = np.maximum(opening[:-1], opening[1:])
            lsv_terms = np.ones_like(diffs, dtype=float)
            non_zero = np.where(max_vals != 0)
            lsv_terms[non_zero] = 1 - (diffs[non_zero] / max_vals[non_zero])
            lsv = np.mean(lsv_terms)

            aav = aav_scores[i]
            sms = aav * lsv

            beam_segment_details.append({
                'BeamNumber': beam.BeamNumber,
                'BeamName': beam.get("BeamName", f"Beam {beam.BeamNumber}"),
                'ControlPointIndex': non_zero_idx[i],
                'CumulativeMetersetWeight_at_CP': mu_weights[non_zero_idx[i]],
                'SegmentMetersetWeight': segment_weights[i],
                'NormalizedSegmentWeight': norm_weights[i],
                'LeafOpenings_Min': np.min(opening),
                'LeafOpenings_Max': np.max(opening),
                'LeafOpenings_Mean': np.mean(opening),
                'SegmentArea_SumOfOpenings': np.sum(opening),
                'LSV': lsv,
                'AAV': aav,
                'SMS': sms
            })

        sms_values = np.array([s['SMS'] for s in beam_segment_details])
        mcs_beam = np.sum(sms_values * norm_weights)
        beam_mcs_scores.append(mcs_beam)
        all_segment_details.extend(beam_segment_details)

    if beam_mcs_scores:
        plan_mcs = float(np.mean(beam_mcs_scores))
        return plan_mcs, all_segment_details
    return None, []


if __name__ == "__main__":
    # Replace this path with your RTPLAN DICOM file path
    rtplan_file = r"s:\ProKnow\Veszprem\Tudo\RP_Tüdő_def_2.dcm"
    output_csv = "mcs_segment_details.csv"

    if os.path.exists(rtplan_file):
        mcs_value, segment_details = calculate_mcs_from_rtplan(rtplan_file)
        if segment_details:
            df = pd.DataFrame(segment_details)
            df = df.round(4)
            df.to_csv(output_csv, index=False)
            print(f"MCS: {mcs_value:.4f}" if mcs_value is not None else "MCS calculation failed.")
            print(f"Detailed segment data saved to: {output_csv}")
        else:
            print("No segment details found for MCS calculation.")
    else:
        print(f"RTPLAN file not found: {rtplan_file}")
