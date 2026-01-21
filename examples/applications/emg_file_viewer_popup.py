"""
Lightweight wrapper that preserves the original `emg_file_viewer.py` and adds
a Tools -> 3D Cone menu action to launch the Plotly popup (`test_3d_cone_popup.py`).

Run this file instead of `emg_file_viewer.py` to get the integrated popup.
"""
import sys
import os
import numpy as np
import tempfile
import plotly.graph_objects as go
import json
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings, QWebEnginePage
from PyQt5.QtCore import QUrl

try:
    import emg_file_viewer as ev
except Exception as e:
    raise ImportError(f"Could not import emg_file_viewer: {e}")

QtWidgets = getattr(ev, 'QtWidgets', None)

def launch_3d_cone(viewer, intensities=None, colormap='Viridis'):
    # Robustly load the test_3d_cone_popup module by path (works regardless
    # of current working directory / import path issues) and surface errors.
    try:
        popup_path = os.path.join(os.path.dirname(__file__), 'test_3d_cone_popup.py')
        if not os.path.exists(popup_path):
            raise FileNotFoundError(popup_path)
        import importlib.util
        spec = importlib.util.spec_from_file_location('test_3d_cone_popup', popup_path)
        popup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(popup)
    except Exception as ex:
        # show error dialog and print to console for debugging
        try:
            print('Failed to load test_3d_cone_popup:', ex)
            QtWidgets.QMessageBox.critical(viewer, '3D Cone Load Error', str(ex))
        except Exception:
            print('UI unavailable, error:', ex)
        return

    try:
        html = popup.generate_plotly_3d_cone(intensities=intensities, colormap=colormap)
    except Exception as ex:
        print('generate_plotly_3d_cone failed:', ex)
        try:
            QtWidgets.QMessageBox.critical(viewer, '3D Cone Error', str(ex))
        except Exception:
            pass
        return

    try:
        # keep reference on viewer to avoid GC closing the window
        viewer._plotly_popup = popup.PlotlyWindow(html)
        viewer._plotly_popup.show()
        try:
            viewer._plotly_popup.raise_()
            viewer._plotly_popup.activateWindow()
        except Exception:
            pass
    except Exception as ex:
        print('Failed to create/show PlotlyWindow:', ex)
        try:
            QtWidgets.QMessageBox.critical(viewer, '3D Cone Error', str(ex))
        except Exception:
            pass

def main():
    if not getattr(ev, 'GUI_AVAILABLE', False) or not getattr(ev, 'MATPLOTLIB_AVAILABLE', False):
        # fallback to original main which handles headless
        ev.main()
        return

    # Ensure Qt attribute AA_ShareOpenGLContexts is set before creating QApplication
    try:
        from PyQt5 import QtCore as _qtcore
        _qtcore.QCoreApplication.setAttribute(_qtcore.Qt.AA_ShareOpenGLContexts, True)
    except Exception:
        try:
            # fallback to ev.QtCore if available
            if getattr(ev, 'QtCore', None) is not None:
                ev.QtCore.QCoreApplication.setAttribute(ev.QtCore.Qt.AA_ShareOpenGLContexts, True)
        except Exception:
            pass

    app = QtWidgets.QApplication(sys.argv)
    viewer = ev.EMGViewer()

    # Replace the RMS matplotlib canvas with an embedded Plotly QWebEngineView
    try:
        # locate the existing canvas2 widget and its parent/layout
        old_canvas = getattr(viewer, 'canvas2', None)
        if old_canvas is not None:
            parent = old_canvas.parent()
            layout = parent.layout() if parent is not None else None
            # create web view
            viewer._rms_web = QWebEngineView()
            # ensure the web view expands to fill available space
            try:
                viewer._rms_web.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                viewer._rms_web.setContentsMargins(0, 0, 0, 0)
            except Exception:
                pass
            # attach a page that prints JS console messages for easier debugging
            class PageWithConsole(QWebEnginePage):
                def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
                    print(f"RMS Web JS console ({level}) {sourceID}:{lineNumber} - {message}")

            page = PageWithConsole(viewer._rms_web)
            viewer._rms_web.setPage(page)
            # enable needed settings
            try:
                settings = viewer._rms_web.settings()
                settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
                settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
                settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
            except Exception:
                pass
            if layout is not None:
                idx = layout.indexOf(old_canvas)
                # remove old canvas
                try:
                    layout.removeWidget(old_canvas)
                    old_canvas.hide()
                except Exception:
                    pass
                # insert web view at same index with stretch so it fills panel
                try:
                    layout.insertWidget(idx, viewer._rms_web, 1)
                    layout.setStretch(idx, 1)
                except Exception:
                    layout.insertWidget(idx, viewer._rms_web)
            else:
                # fallback: add to viewer.canvas2's parent
                try:
                    parent_layout = viewer.canvas2.parent().layout()
                    parent_layout.addWidget(viewer._rms_web, 1)
                    try:
                        parent_layout.setStretch(parent_layout.indexOf(viewer._rms_web), 1)
                    except Exception:
                        pass
                except Exception:
                    pass
    except Exception:
        viewer._rms_web = None

    def _render_plotly_rms(grid, rows, cols, cmap='viridis'):
        """Render a 2D Plotly heatmap into the embedded RMS web view.

        Uses Plotly.react to update the existing plot in-place for speed.
        """
        if getattr(viewer, '_rms_web', None) is None:
            return

        # create an initial HTML with embedded Plotly once
            # create a small stable HTML shell once that loads Plotly from CDN
            if not getattr(viewer, '_rms_html_path', None):
                tmp = tempfile.gettempdir()
                fp0 = os.path.join(tmp, 'rms_plotly_shell.html')
                shell_html = '''<!doctype html>
    <html><head><meta charset="utf-8"><title>RMS Plot</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    </head><body>
    <div id="plot" class="js-plotly-plot" style="width:100%;height:100%;"></div>
    <script>
    function notifyReady(){ if(window.Plotly){window._plotly_ready = true;} else {setTimeout(notifyReady,50);} }
    notifyReady();
    window.updatePlot = function(payload){
        try{
            var p = payload;
            if(typeof p === 'string') p = JSON.parse(p);
            if(!p) return;
            var gd = document.querySelector('.js-plotly-plot');
            if(!gd || typeof Plotly === 'undefined') return;
            if(p.action === 'heatmap'){
                try{ Plotly.react(gd, p.data, p.layout, p.config); return; }catch(e){/*fallback*/}
            } else if(p.action === '3d'){
                try{ Plotly.newPlot(gd, p.data, p.layout, p.config); return; }catch(e){/*fallback*/}
            }
        }catch(e){console.error('updatePlot error', e);}    
    }
    </script>
    </body></html>'''
                try:
                    with open(fp0, 'w', encoding='utf-8') as f:
                        f.write(shell_html)
                except Exception:
                    pass
                viewer._rms_html_path = fp0
                viewer._rms_web.load(QUrl.fromLocalFile(fp0))

        # map common cmap names to Plotly-compatible colorscales
        try:
            cmap_map = {'viridis': 'Viridis', 'inferno': 'Inferno', 'magma': 'Magma', 'plasma': 'Plasma', 'cividis': 'Cividis', 'jet': 'Jet'}
            cmap_plotly = cmap_map.get(str(cmap).lower(), cmap)
        except Exception:
            cmap_plotly = cmap

        try:
            data_obj = [{
                'z': grid.tolist(),
                'type': 'heatmap',
                'colorscale': cmap_plotly,
                'showscale': True
            }]
            layout_obj = {'title': 'RMS heatmap', 'xaxis': {'title': 'Col'}, 'yaxis': {'title': 'Row'}}
            config_obj = {'displayModeBar': True, 'responsive': True, 'scrollZoom': True}
            payload = {'action': 'heatmap', 'data': data_obj, 'layout': layout_obj, 'config': config_obj}
            # call updatePlot but wait until the function exists in the page
            # store the payload object and send it to the page once ready
            viewer._rms_pending_payload = payload

            # attach loadFinished handler once so we only run JS when Plotly is available
            if not getattr(viewer, '_rms_load_connected', False):
                def _on_rms_loaded(ok):
                    viewer._rms_page_ready = bool(ok)
                    # start a short polling timer to wait until Plotly (from CDN) is ready
                    try:
                        if getattr(viewer, '_rms_ready_timer', None) is None:
                            viewer._rms_ready_timer = ev.QtCore.QTimer()
                            viewer._rms_ready_timer.setInterval(100)

                            def _poll_ready():
                                try:
                                    def _cb(res):
                                        try:
                                            if bool(res):
                                                # create an initial plot so the plot area is visible
                                                try:
                                                    init_js = "Plotly.newPlot(document.querySelector('.js-plotly-plot'), [{z:[[0]],type:'heatmap'}], {title:'RMS heatmap'}, {displayModeBar:true});"
                                                    viewer._rms_web.page().runJavaScript(init_js)
                                                except Exception:
                                                    pass
                                                # run pending payload if present
                                                if getattr(viewer, '_rms_pending_payload', None):
                                                    try:
                                                        payload_to_send = viewer._rms_pending_payload
                                                        js_call_p = "(function(p){ try{ if(typeof window.updatePlot==='function'){ window.updatePlot(p); } else { Plotly.newPlot(document.querySelector('.js-plotly-plot'), p.data, p.layout, p.config); } } catch(e){ console.error('updatePlot error', e); } })(%s);" % json.dumps(payload_to_send)
                                                        viewer._rms_web.page().runJavaScript(js_call_p)
                                                    except Exception:
                                                        pass
                                                    viewer._rms_pending_payload = None
                                                try:
                                                    viewer._rms_ready_timer.stop()
                                                except Exception:
                                                    pass

                                        except Exception:
                                            pass
                                    viewer._rms_web.page().runJavaScript("(typeof window._plotly_ready !== 'undefined') && window._plotly_ready", _cb)
                                except Exception:
                                    pass

                            viewer._rms_ready_timer.timeout.connect(_poll_ready)
                        try:
                            viewer._rms_ready_timer.start()
                        except Exception:
                            pass
                    except Exception:
                        pass

                try:
                    viewer._rms_web.page().loadFinished.connect(_on_rms_loaded)
                    viewer._rms_load_connected = True
                except Exception:
                    pass

            # if page is ready, send the payload JSON to the page's updatePlot function
            if getattr(viewer, '_rms_page_ready', False):
                try:
                    dump_path = os.path.join(os.path.dirname(__file__), '_last_rms_payload.json')
                    with open(dump_path, 'w', encoding='utf-8') as df:
                        json.dump(payload, df)
                except Exception:
                    pass
                try:
                    js_call = "(function(p){ try{ if(typeof window.updatePlot==='function'){ window.updatePlot(p); } else { Plotly.newPlot(document.querySelector('.js-plotly-plot'), p.data, p.layout, p.config); } } catch(e){ console.error('updatePlot error', e); } })(%s);" % json.dumps(payload)
                    # payload sent (debug print removed)
                    viewer._rms_web.page().runJavaScript(js_call)
                except Exception as e:
                    print('runJavaScript failed for heatmap payload:', e)
                    print('Heatmap payload snippet:', str(payload)[:1000])
            else:
                viewer._rms_pending_payload = payload
        except Exception as ex:
            print('Failed to update heatmap via JS:', ex)

    def _render_plotly_3d(intensities, colormap='Viridis'):
        """Render the 3D cone in the embedded RMS web view using generated coords and Plotly.react."""
        if getattr(viewer, '_rms_web', None) is None:
            return
        try:
            num_channels = 128
            num_columns = 13
            num_rows = num_channels // num_columns
            theta = np.linspace(0, 2 * np.pi, num_columns, endpoint=False)
            z_vals = np.linspace(0, 2.0, num_rows)
            radii = np.linspace(1.0, 0.5, num_rows)
            x_points = []
            y_points = []
            z_points = []
            for i, z_val in enumerate(z_vals):
                for j, theta_val in enumerate(theta):
                    x_points.append(radii[i] * np.cos(theta_val))
                    y_points.append(radii[i] * np.sin(theta_val))
                    z_points.append(z_val)

            # ensure initial HTML loaded
                # ensure shell HTML loaded (use the shared shell created by _render_plotly_rms)
                if not getattr(viewer, '_rms_html_path', None):
                    tmp = tempfile.gettempdir()
                    fp0 = os.path.join(tmp, 'rms_plotly_shell.html')
                    # if the shared shell isn't present, create a robust fallback (same as the main shell)
                    if not os.path.exists(fp0):
                        shell_html = '''<!doctype html>
    <html><head><meta charset="utf-8"><title>RMS Plot</title>
    <meta http-equiv="Content-Security-Policy" content="default-src * 'unsafe-inline' 'unsafe-eval' data:;"></meta>
    <style>html,body,#plot{height:100%;width:100%;margin:0;padding:0;}#plot{display:block;}</style>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    </head><body>
    <div id="plot" class="js-plotly-plot"></div>
    <script>
    window._plotly_ready = false;
    (function waitForPlotly(){ if(window.Plotly){ window._plotly_ready = true; } else { setTimeout(waitForPlotly,50); } })();
    window.updatePlot = function(payload){
        try{
            if(!payload) return;
            if(payload.action === 'heatmap' || payload.action === '3d'){
                Plotly.newPlot(document.querySelector('.js-plotly-plot'), payload.data, payload.layout, payload.config).catch(function(e){ console.error('Plotly.newPlot failed', e); });
            } else {
                try{ Plotly.react(document.querySelector('.js-plotly-plot'), payload.data, payload.layout, payload.config); }catch(e){ console.error('Plotly.react failed', e); }
            }
        }catch(err){ console.error('updatePlot error', err); }
    };
    window.addEventListener('error', function(ev){ console.error('window.onerror', ev.message, ev.error); });
    </script>
    </body></html>'''
                        try:
                            with open(fp0, 'w', encoding='utf-8') as f:
                                f.write(shell_html)
                        except Exception:
                            pass
                    viewer._rms_html_path = fp0
                    viewer._rms_web.load(QUrl.fromLocalFile(fp0))
                # write debug copy
                try:
                    repo_debug = os.path.join(os.path.dirname(__file__), '_rms_plotly_debug.html')
                    with open(fp0, 'r', encoding='utf-8') as src, open(repo_debug, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                    with open(repo_debug, 'r', encoding='utf-8') as fdbg:
                        s = fdbg.read()
                    if s.count('(') != s.count(')') or s.count('{') != s.count('}'):
                        print('Debug HTML has unbalanced parens/braces: (', s.count('('), ')', s.count(')'), ' {', s.count('{'), ' }', s.count('}'))
                except Exception:
                    pass

            # map colormap
            try:
                cmap_map = {'viridis': 'Viridis', 'inferno': 'Inferno', 'magma': 'Magma', 'plasma': 'Plasma', 'cividis': 'Cividis', 'jet': 'Jet'}
                cmap_plotly = cmap_map.get(str(colormap).lower(), colormap)
            except Exception:
                cmap_plotly = colormap

            # update marker colors via Plotly.react
            try:
                data_obj = [{
                    'x': x_points,
                    'y': y_points,
                    'z': z_points,
                    'mode': 'markers',
                    'type': 'scatter3d',
                    'marker': {'size': 5, 'color': intensities, 'colorscale': cmap_plotly}
                }]
                layout_obj = {'title': '3D Cone Layout (Plotly)'}
                config_obj = {'displayModeBar': True, 'responsive': True, 'scrollZoom': True}
                # Use the page's updatePlot function to perform a safe newPlot for 3D
                payload = {'action': '3d', 'data': data_obj, 'layout': layout_obj, 'config': config_obj}
                # store pending payload and send when ready
                viewer._rms_pending_payload = payload

                # attach loadFinished handler once so we only run payload when Plotly is available
                if not getattr(viewer, '_rms_load_connected', False):
                    def _on_rms_loaded(ok):
                        viewer._rms_page_ready = bool(ok)
                        # if there's a pending payload, send it safely
                        try:
                            if getattr(viewer, '_rms_pending_payload', None):
                                payload_to_send = viewer._rms_pending_payload
                                js_call = "(function(p){ try{ if(typeof window.updatePlot==='function'){ window.updatePlot(p); } else { Plotly.newPlot(document.querySelector('.js-plotly-plot'), p.data, p.layout, p.config); } } catch(e){ console.error('updatePlot error', e); } })(%s);" % json.dumps(payload_to_send)
                                viewer._rms_web.page().runJavaScript(js_call)
                                viewer._rms_pending_payload = None
                        except Exception:
                            pass

                    try:
                        page = viewer._rms_web.page()
                        page.loadFinished.connect(_on_rms_loaded)
                        # try to forward console messages to Python for visibility
                        try:
                            def _console_msg(level, message, line, source):
                                try:
                                    print('RMS Web JS console (level %s) %s:%s - %s' % (level, source, line, message))
                                except Exception:
                                    print('RMS Web JS console:', message)
                            page.javaScriptConsoleMessage.connect(_console_msg)
                        except Exception:
                            pass
                        viewer._rms_load_connected = True
                    except Exception:
                        pass

                if getattr(viewer, '_rms_page_ready', False):
                    try:
                        try:
                            dump_path = os.path.join(os.path.dirname(__file__), '_last_rms_payload.json')
                            with open(dump_path, 'w', encoding='utf-8') as df:
                                json.dump(payload, df)
                        except Exception:
                            pass
                        js_call = "(function(p){ try{ if(typeof window.updatePlot==='function'){ window.updatePlot(p); } else { Plotly.newPlot(document.querySelector('.js-plotly-plot'), p.data, p.layout, p.config); } } catch(e){ console.error('updatePlot error', e); } })(%s);" % json.dumps(payload)
                        # payload sent (debug print removed)
                        viewer._rms_web.page().runJavaScript(js_call)
                    except Exception as e:
                        print('runJavaScript failed for 3D payload:', e)
                        print('3D payload snippet:', str(payload)[:1000])
                else:
                    viewer._rms_pending_payload = payload
            except Exception as ex:
                print('Failed to update 3D via JS:', ex)
        except Exception as ex:
            print('Failed to render 3D in embedded view:', ex)

    # Create a debounced renderer to avoid frequent heavy updates
    try:
        viewer._rms_update_timer = ev.QtCore.QTimer()
        viewer._rms_update_timer.setSingleShot(True)
        viewer._rms_update_timer.setInterval(200)  # ms debounce
    except Exception:
        viewer._rms_update_timer = None

    def _do_rms_render():
        try:
            # compute rms_mean similarly to patched_plot_heatmap previous logic
            if getattr(viewer, 'data', None) is None:
                return
            try:
                fs = float(max(1.0, float(viewer.spin_fs.value())))
            except Exception:
                fs = float(getattr(viewer, 'fs', 2000.0))
            try:
                rms_ms = int(getattr(viewer, 'spin_rms_ms').value()) if getattr(viewer, 'spin_rms_ms', None) is not None else 100
                win_samps = max(1, int((rms_ms / 1000.0) * fs))
                center = int(getattr(viewer, 'current_pos', 0))
                start = max(0, center - win_samps // 2)
                end = min(viewer.data.shape[0], start + win_samps)
                if viewer.chk_apply_filters_to_rms.isChecked() and ((getattr(viewer, 'chk_bp_enable', None) and viewer.chk_bp_enable.isChecked()) or (getattr(viewer, 'chk_notch_enable', None) and viewer.chk_notch_enable.isChecked())):
                    try:
                        block = viewer.data[start:end, :]
                        if block.shape[0] < 1:
                            arr = viewer.data
                        else:
                            arr = block
                        fs = float(max(1.0, float(viewer.spin_fs.value())))
                        n_ch = int(viewer.data.shape[1])
                        notch_hz = float(getattr(viewer, 'spin_notch_hz_vis', None).value()) if getattr(viewer, 'spin_notch_hz_vis', None) is not None else None
                        notch_q = float(getattr(viewer, 'spin_qc_notchQ', None).value()) if getattr(viewer, 'spin_qc_notchQ', None) is not None else 30.0
                        bp_low = float(getattr(viewer, 'spin_bp_low_vis', None).value()) if getattr(viewer, 'spin_bp_low_vis', None) is not None else 10.0
                        bp_high = float(getattr(viewer, 'spin_bp_high_vis', None).value()) if getattr(viewer, 'spin_bp_high_vis', None) is not None else (fs/2.0 - 1.0)
                        visf = ev._make_vis_filter(fs, n_ch, bp_low, bp_high, notch_hz, notch_q, getattr(viewer, 'chk_bp_enable').isChecked(), getattr(viewer, 'chk_notch_enable').isChecked())
                        if visf is not None:
                            try:
                                proc = visf.process(arr.T)
                                arr2 = proc.T
                            except Exception:
                                arr2 = arr
                        else:
                            arr2 = arr
                        if arr2.shape[0] < 1:
                            rms_mean = np.sqrt(np.mean(viewer.data**2, axis=0))
                        else:
                            rms_mean = np.sqrt(np.mean(arr2**2, axis=0))
                    except Exception:
                        rms_mean = np.sqrt(np.mean(viewer.data**2, axis=0))
                else:
                    if end - start < 1:
                        rms_mean = np.sqrt(np.mean(viewer.data**2, axis=0))
                    else:
                        rms_mean = np.sqrt(np.mean(viewer.data[start:end, :]**2, axis=0))
            except Exception:
                rms_mean = np.sqrt(np.mean(viewer.data**2, axis=0))

            # Prepare grid
            try:
                layout_text = str(viewer.combo_layout.currentText()) if getattr(viewer, 'combo_layout', None) is not None else ''
                if layout_text == 'Custom':
                    rows = int(viewer.spin_rows.value())
                    cols = int(viewer.spin_cols.value())
                else:
                    if layout_text == '8x8': rows, cols = 8, 8
                    elif layout_text == '4x16': rows, cols = 4, 16
                    elif layout_text == '16x4': rows, cols = 16, 4
                    elif layout_text == '3D Cone': rows, cols = 8, 16
                    else: rows, cols = 8, 8
            except Exception:
                rows, cols = 8, 8

            nch = viewer.data.shape[1]
            grid = np.full((rows, cols), np.nan)
            for i in range(min(nch, rows * cols)):
                r = i // cols
                c = i % cols
                try:
                    if getattr(viewer, 'qc_instance', None) is not None and viewer.qc_instance._is_bad[i]:
                        grid[r, c] = np.nan
                    else:
                        grid[r, c] = float(rms_mean[i])
                except Exception:
                    grid[r, c] = float(rms_mean[i]) if i < len(rms_mean) else np.nan

            if layout_text == '3D Cone':
                _render_plotly_3d(rms_mean.tolist(), colormap=str(viewer.cmap_combo.currentText()) if getattr(viewer, 'cmap_combo', None) is not None else 'Viridis')
            else:
                _render_plotly_rms(grid, rows, cols, cmap=str(viewer.cmap_combo.currentText()) if getattr(viewer, 'cmap_combo', None) is not None else 'viridis')
        except Exception as ex:
            print('Error in _do_rms_render:', ex)

    # connect timer to render function
    try:
        if viewer._rms_update_timer is not None:
            viewer._rms_update_timer.timeout.connect(_do_rms_render)
    except Exception:
        pass

    # Monkeypatch plot_heatmap so that when layout == '3D Cone' we compute
    # RMS values and open the Plotly popup with those intensities.
    try:
        original_plot_heatmap = viewer.plot_heatmap

        def patched_plot_heatmap(*a, **kw):
            try:
                original_plot_heatmap(*a, **kw)
            except Exception:
                # still attempt to render via Plotly even if original plotting fails
                pass

            try:
                layout = str(viewer.combo_layout.currentText()) if getattr(viewer, 'combo_layout', None) is not None else ''
                # compute rms_mean (shared for 2D and 3D)
                if getattr(viewer, 'data', None) is None:
                    return
                try:
                    fs = float(max(1.0, float(viewer.spin_fs.value())))
                except Exception:
                    fs = float(getattr(viewer, 'fs', 2000.0))
                try:
                    rms_ms = int(getattr(viewer, 'spin_rms_ms').value()) if getattr(viewer, 'spin_rms_ms', None) is not None else 100
                    win_samps = max(1, int((rms_ms / 1000.0) * fs))
                    center = int(getattr(viewer, 'current_pos', 0))
                    start = max(0, center - win_samps // 2)
                    end = min(viewer.data.shape[0], start + win_samps)
                    if viewer.chk_apply_filters_to_rms.isChecked() and ((getattr(viewer, 'chk_bp_enable', None) and viewer.chk_bp_enable.isChecked()) or (getattr(viewer, 'chk_notch_enable', None) and viewer.chk_notch_enable.isChecked())):
                        try:
                            block = viewer.data[start:end, :]
                            if block.shape[0] < 1:
                                arr = viewer.data
                            else:
                                arr = block
                            fs = float(max(1.0, float(viewer.spin_fs.value())))
                            n_ch = int(viewer.data.shape[1])
                            notch_hz = float(getattr(viewer, 'spin_notch_hz_vis', None).value()) if getattr(viewer, 'spin_notch_hz_vis', None) is not None else None
                            notch_q = float(getattr(viewer, 'spin_qc_notchQ', None).value()) if getattr(viewer, 'spin_qc_notchQ', None) is not None else 30.0
                            bp_low = float(getattr(viewer, 'spin_bp_low_vis', None).value()) if getattr(viewer, 'spin_bp_low_vis', None) is not None else 10.0
                            bp_high = float(getattr(viewer, 'spin_bp_high_vis', None).value()) if getattr(viewer, 'spin_bp_high_vis', None) is not None else (fs/2.0 - 1.0)
                            visf = ev._make_vis_filter(fs, n_ch, bp_low, bp_high, notch_hz, notch_q, getattr(viewer, 'chk_bp_enable').isChecked(), getattr(viewer, 'chk_notch_enable').isChecked())
                            if visf is not None:
                                try:
                                    proc = visf.process(arr.T)
                                    arr2 = proc.T
                                except Exception:
                                    arr2 = arr
                            else:
                                arr2 = arr
                            if arr2.shape[0] < 1:
                                rms_mean = np.sqrt(np.mean(viewer.data**2, axis=0))
                            else:
                                rms_mean = np.sqrt(np.mean(arr2**2, axis=0))
                        except Exception:
                            rms_mean = np.sqrt(np.mean(viewer.data**2, axis=0))
                    else:
                        if end - start < 1:
                            rms_mean = np.sqrt(np.mean(viewer.data**2, axis=0))
                        else:
                            rms_mean = np.sqrt(np.mean(viewer.data[start:end, :]**2, axis=0))
                except Exception:
                    rms_mean = np.sqrt(np.mean(viewer.data**2, axis=0))

                # Prepare grid for 2D layouts
                try:
                    layout_text = str(viewer.combo_layout.currentText()) if getattr(viewer, 'combo_layout', None) is not None else ''
                    if layout_text == 'Custom':
                        rows = int(viewer.spin_rows.value())
                        cols = int(viewer.spin_cols.value())
                    else:
                        if layout_text == '8x8': rows, cols = 8, 8
                        elif layout_text == '4x16': rows, cols = 4, 16
                        elif layout_text == '16x4': rows, cols = 16, 4
                        elif layout_text == '3D Cone': rows, cols = 8, 16
                        else: rows, cols = 8, 8
                except Exception:
                    rows, cols = 8, 8

                nch = viewer.data.shape[1]
                grid = np.full((rows, cols), np.nan)
                for i in range(min(nch, rows * cols)):
                    r = i // cols
                    c = i % cols
                    try:
                        if getattr(viewer, 'qc_instance', None) is not None and viewer.qc_instance._is_bad[i]:
                            grid[r, c] = np.nan
                        else:
                            grid[r, c] = float(rms_mean[i])
                    except Exception:
                        grid[r, c] = float(rms_mean[i]) if i < len(rms_mean) else np.nan

                # Render into embedded view: 3D cone or 2D heatmap
                try:
                    if layout == '3D Cone':
                        _render_plotly_3d(rms_mean.tolist(), colormap=str(viewer.cmap_combo.currentText()) if getattr(viewer, 'cmap_combo', None) is not None else 'Viridis')
                    else:
                        _render_plotly_rms(grid, rows, cols, cmap=str(viewer.cmap_combo.currentText()) if getattr(viewer, 'cmap_combo', None) is not None else 'viridis')
                except Exception as ex:
                    print('Failed to render Plotly RMS:', ex)
            except Exception as ex:
                print('Error in patched_plot_heatmap:', ex)

        viewer.plot_heatmap = patched_plot_heatmap
        # Also trigger plot_heatmap when layout selection changes
        try:
            viewer.combo_layout.currentIndexChanged.connect(lambda _idx: viewer.plot_heatmap())
        except Exception:
            pass
        # ensure changing colormap triggers redraw
        try:
            if getattr(viewer, 'cmap_combo', None) is not None:
                viewer.cmap_combo.currentIndexChanged.connect(lambda _idx: viewer.plot_heatmap())
        except Exception:
            pass
    except Exception:
        pass

    # Ensure RMS tab activation triggers a redraw into the embedded view
    try:
        def on_tab_changed(idx):
            # RMS Heatmap tab is index 1 in the original UI
            try:
                if getattr(viewer, 'plot_tabs', None) is not None and idx == 1:
                    viewer.plot_heatmap()
            except Exception:
                pass

        viewer.plot_tabs.currentChanged.connect(on_tab_changed)
    except Exception:
        pass

    # Try an initial render for RMS panel if data already loaded
    try:
        viewer.plot_heatmap()
    except Exception:
        pass

    # Add Tools menu with 3D Cone action
    try:
        tools_menu = viewer.menuBar().addMenu('Tools')
        act = QtWidgets.QAction('3D Cone', viewer)
        act.triggered.connect(lambda: launch_3d_cone(viewer))
        tools_menu.addAction(act)
    except Exception:
        pass

    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
