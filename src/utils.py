def create_output_json(patient_id, Cm_img, Cm_tab, Cm_note, Rf, final_label, Sc=None, issues=[]):
    return {
        "patient_id": patient_id,
        "Cm_img": float(Cm_img),
        "Cm_tab": float(Cm_tab),
        "Cm_note": float(Cm_note),
        "Sc": float(Sc) if Sc is not None else None,
        "Rf": float(Rf),
        "final_label": final_label,
        "QC_flag": Rf < 0.75,
        "issues": issues
    }
