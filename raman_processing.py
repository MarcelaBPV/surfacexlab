le "/mount/src/surfacexlab/raman_tab.py", line 118, in render_raman_tab
    result = process_raman_spectrum_with_groups(
        file
    )
File "/mount/src/surfacexlab/raman_processing.py", line 170, in process_raman_spectrum_with_groups
    y_smooth = savgol_filter(
    
    ...<4 lines>...
        polyorder=3
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/scipy/signal/_savitzky_golay.py", line 371, in savgol_filter
    raise ValueError("If mode is 'interp', window_length must be less "
                     "than or equal to the size of x.")
