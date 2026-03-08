#!/usr/bin/env python3
"""
Dataset Builder GUI - Enhanced Configuration Interface

A comprehensive GUI for building EMG gesture classification datasets with
full control over all processing parameters, feature extraction, and file handling.

Features:
- Root directory selection with file discovery
- Flexible file type support (RHD, NPZ, CSV)
- Configurable feature extraction parameters
- Visual display of processing pipeline steps
- Event file management
- Channel selection and mapping
- Real-time parameter validation
- Profile save/load

Usage:
    python dataset_builder_gui.py
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QTextEdit, QFileDialog, QTabWidget,
    QScrollArea, QGridLayout, QListWidget, QSplitter, QFrame,
    QMessageBox, QProgressBar
)

# Try to import intan package to check available features
try:
    from intan.processing import FEATURE_REGISTRY
    AVAILABLE_FEATURES = list(FEATURE_REGISTRY.keys())
except ImportError:
    AVAILABLE_FEATURES = ['mean_absolute_value', 'root_mean_square', 'variance', 
                         'waveform_length', 'zero_crossings', 'slope_sign_changes']


class CollapsibleSection(QWidget):
    """A collapsible section widget with a toggle button."""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_btn = QPushButton(f"▼ {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.setStyleSheet("text-align: left; font-weight: bold;")
        self.toggle_btn.clicked.connect(self.toggle_content)
        
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(20, 5, 0, 5)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.content)
        
    def toggle_content(self, checked):
        self.content.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self.toggle_btn.setText(f"{arrow} {self.toggle_btn.text()[2:]}")
    
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


class FeaturePipelineWidget(QWidget):
    """Visual display of feature extraction pipeline steps."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("<b>Feature Extraction Pipeline</b>")
        title.setStyleSheet("font-size: 14px; color: #0A84FF;")
        layout.addWidget(title)
        
        # Pipeline visualization
        self.pipeline_text = QTextEdit()
        self.pipeline_text.setReadOnly(True)
        self.pipeline_text.setMaximumHeight(200)
        self.pipeline_text.setStyleSheet("""
            QTextEdit {
                background-color: #F6F7F9;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.pipeline_text)
        
        self.update_pipeline()
    
    def update_pipeline(self, params: Optional[Dict] = None):
        """Update pipeline display with current parameters."""
        if params is None:
            params = {}
        
        pipeline_text = []
        pipeline_text.append("📊 <b>Feature Extraction Pipeline:</b><br>")
        pipeline_text.append("━" * 60 + "<br>")
        
        # Step 1: File Loading
        file_type = params.get('file_type', 'rhd').upper()
        pipeline_text.append(f"<b>1. Load Data</b><br>")
        pipeline_text.append(f"   └─ Format: {file_type} files<br>")
        pipeline_text.append(f"   └─ Mode: {'Multi-file aggregation' if params.get('multi_file') else 'Single file'}<br>")
        pipeline_text.append("<br>")
        
        # Step 2: Channel Selection
        pipeline_text.append(f"<b>2. Channel Selection</b><br>")
        if params.get('channel_map'):
            pipeline_text.append(f"   └─ Mapping: {params['channel_map']}<br>")
        elif params.get('channels'):
            pipeline_text.append(f"   └─ Channels: {params['channels']}<br>")
        else:
            pipeline_text.append(f"   └─ Channels: All available<br>")
        pipeline_text.append("<br>")
        
        # Step 3: Preprocessing
        pipeline_text.append(f"<b>3. Signal Preprocessing</b><br>")
        if params.get('paper_style'):
            pipeline_text.append(f"   └─ Mode: Paper-style (120Hz highpass)<br>")
        else:
            pipeline_text.append(f"   └─ Bandpass: 20-500 Hz<br>")
            pipeline_text.append(f"   └─ Notch: 60 Hz<br>")
        pipeline_text.append("<br>")
        
        # Step 4: Windowing
        window_ms = params.get('window_ms', 200)
        step_ms = params.get('step_ms', 50)
        overlap_pct = int((1 - step_ms/window_ms) * 100) if window_ms > 0 else 0
        pipeline_text.append(f"<b>4. Sliding Window</b><br>")
        pipeline_text.append(f"   └─ Window: {window_ms} ms<br>")
        pipeline_text.append(f"   └─ Step: {step_ms} ms ({overlap_pct}% overlap)<br>")
        pipeline_text.append("<br>")
        
        # Step 5: Feature Extraction
        features = params.get('features', AVAILABLE_FEATURES)
        pipeline_text.append(f"<b>5. Feature Extraction</b><br>")
        pipeline_text.append(f"   └─ Features per channel:<br>")
        for feat in features:
            pipeline_text.append(f"      • {feat}<br>")
        pipeline_text.append(f"   └─ Total features/window: {len(features)} × n_channels<br>")
        pipeline_text.append("<br>")
        
        # Step 6: Label Matching
        pipeline_text.append(f"<b>6. Label Matching</b><br>")
        pipeline_text.append(f"   └─ Source: Event files (.txt)<br>")
        ignore_labels = params.get('ignore_labels', ['Start', 'End', 'None'])
        pipeline_text.append(f"   └─ Ignored labels: {', '.join(ignore_labels)}<br>")
        pipeline_text.append("<br>")
        
        # Step 7: Output
        pipeline_text.append(f"<b>7. Save Dataset</b><br>")
        pipeline_text.append(f"   └─ Format: NPZ (NumPy compressed)<br>")
        pipeline_text.append(f"   └─ Contains: X (features), y (labels), metadata<br>")
        
        self.pipeline_text.setHtml("".join(pipeline_text))


class DatasetBuilderGUI(QMainWindow):
    """Main window for dataset builder configuration."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Dataset Builder - Configuration")
        self.setGeometry(100, 100, 1000, 800)
        
        # Initialize state
        self.root_dir = None
        self.discovered_files = []
        self.event_files = []
        
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        """Initialize the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Configuration
        config_widget = self.create_config_panel()
        splitter.addWidget(config_widget)
        
        # Right side: Pipeline visualization
        pipeline_widget = self.create_pipeline_panel()
        splitter.addWidget(pipeline_widget)
        
        splitter.setSizes([600, 400])
        main_layout.addWidget(splitter)
        
        # Bottom: Action buttons
        button_layout = self.create_action_buttons()
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_config_panel(self):
        """Create the left configuration panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("<h2>Dataset Configuration</h2>")
        layout.addWidget(title)
        
        # Scroll area for config
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Section 1: Project Setup
        project_section = self.create_project_section()
        scroll_layout.addWidget(project_section)
        
        # Section 2: File Settings
        file_section = self.create_file_section()
        scroll_layout.addWidget(file_section)
        
        # Section 3: Channel Configuration
        channel_section = self.create_channel_section()
        scroll_layout.addWidget(channel_section)
        
        # Section 4: Processing Parameters
        processing_section = self.create_processing_section()
        scroll_layout.addWidget(processing_section)
        
        # Section 5: Feature Selection
        feature_section = self.create_feature_section()
        scroll_layout.addWidget(feature_section)
        
        # Section 6: Label Filtering
        label_section = self.create_label_section()
        scroll_layout.addWidget(label_section)
        
        # Section 7: Advanced Options
        advanced_section = self.create_advanced_section()
        scroll_layout.addWidget(advanced_section)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return widget
    
    def create_pipeline_panel(self):
        """Create the right pipeline visualization panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.pipeline_widget = FeaturePipelineWidget()
        layout.addWidget(self.pipeline_widget)
        
        # Data files panel
        data_group = QGroupBox("📊 Data Files")
        data_layout = QVBoxLayout()
        
        data_info = QLabel("Raw recordings from 'raw/' directory:")
        data_info.setStyleSheet("color: #6B7280; font-size: 11px;")
        data_layout.addWidget(data_info)
        
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(200)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #F0F9FF;
                border: 1px solid #BAE6FD;
                border-radius: 6px;
            }
        """)
        data_layout.addWidget(self.file_list)
        
        self.data_count_label = QLabel("No files discovered yet")
        self.data_count_label.setStyleSheet("color: #0369A1; font-weight: bold; font-size: 11px;")
        data_layout.addWidget(self.data_count_label)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Event files panel (separate and visually distinct)
        event_group = QGroupBox("🏷️ Event Files")
        event_layout = QVBoxLayout()
        
        event_info = QLabel("Label files from 'events/' directory:")
        event_info.setStyleSheet("color: #6B7280; font-size: 11px;")
        event_layout.addWidget(event_info)
        
        self.event_list = QListWidget()
        self.event_list.setMaximumHeight(150)
        self.event_list.setStyleSheet("""
            QListWidget {
                background-color: #FEF3C7;
                border: 1px solid #FDE68A;
                border-radius: 6px;
            }
        """)
        event_layout.addWidget(self.event_list)
        
        self.event_count_label = QLabel("No event files discovered yet")
        self.event_count_label.setStyleSheet("color: #92400E; font-weight: bold; font-size: 11px;")
        event_layout.addWidget(self.event_count_label)
        
        event_group.setLayout(event_layout)
        layout.addWidget(event_group)
        
        return widget
    
    def create_project_section(self):
        """Create project setup section."""
        section = CollapsibleSection("📁 Project Setup")
        
        # Root directory
        root_layout = QHBoxLayout()
        root_layout.addWidget(QLabel("Root Directory:"))
        self.root_dir_edit = QLineEdit()
        self.root_dir_edit.setPlaceholderText("Select project root directory...")
        self.root_dir_edit.textChanged.connect(self.on_root_dir_changed)
        root_layout.addWidget(self.root_dir_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_root_dir)
        root_layout.addWidget(browse_btn)
        section.add_layout(root_layout)
        
        # Label/prefix
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Dataset Label:"))
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Optional prefix for output filename")
        self.label_edit.textChanged.connect(self.update_pipeline_display)
        label_layout.addWidget(self.label_edit)
        section.add_layout(label_layout)
        
        return section
    
    def create_file_section(self):
        """Create file settings section."""
        section = CollapsibleSection("📄 File Settings")
        
        # File type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("File Type:"))
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(['rhd', 'npz', 'csv', 'poly5'])
        self.file_type_combo.currentTextChanged.connect(self.on_file_type_changed)
        type_layout.addWidget(self.file_type_combo)
        type_layout.addStretch()
        section.add_layout(type_layout)
        
        # Multi-file mode
        self.multi_file_check = QCheckBox("Multi-file mode (aggregate multiple recordings)")
        self.multi_file_check.setChecked(True)
        self.multi_file_check.stateChanged.connect(self.update_pipeline_display)
        section.add_widget(self.multi_file_check)
        
        # File patterns
        pattern_layout = QGridLayout()
        pattern_layout.addWidget(QLabel("Include Pattern:"), 0, 0)
        self.merge_pattern_edit = QLineEdit()
        self.merge_pattern_edit.setPlaceholderText("e.g., train_* (optional)")
        pattern_layout.addWidget(self.merge_pattern_edit, 0, 1)
        
        pattern_layout.addWidget(QLabel("Exclude Pattern:"), 1, 0)
        self.exclude_pattern_edit = QLineEdit()
        self.exclude_pattern_edit.setPlaceholderText("e.g., *_test (optional)")
        pattern_layout.addWidget(self.exclude_pattern_edit, 1, 1)
        
        section.add_layout(pattern_layout)
        
        # Refresh files button
        refresh_btn = QPushButton("🔄 Discover Files")
        refresh_btn.clicked.connect(self.discover_files)
        section.add_widget(refresh_btn)
        
        return section
    
    def create_channel_section(self):
        """Create channel configuration section."""
        section = CollapsibleSection("📡 Channel Configuration")
        
        # Channel mapping
        map_layout = QHBoxLayout()
        map_layout.addWidget(QLabel("Channel Map:"))
        self.channel_map_combo = QComboBox()
        self.channel_map_combo.addItems(['None', '8-8-L', '8-8-R', '16-4', 'Custom'])
        self.channel_map_combo.currentTextChanged.connect(self.update_pipeline_display)
        map_layout.addWidget(self.channel_map_combo)
        map_layout.addStretch()
        section.add_layout(map_layout)
        
        # Manual channel selection
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Or specify channels:"))
        self.channels_edit = QLineEdit()
        self.channels_edit.setPlaceholderText("e.g., 0:64 or 0,5,10-20")
        self.channels_edit.textChanged.connect(self.update_pipeline_display)
        channel_layout.addWidget(self.channels_edit)
        section.add_layout(channel_layout)
        
        # Non-strict mapping
        self.mapping_non_strict_check = QCheckBox("Allow missing channels (non-strict mapping)")
        section.add_widget(self.mapping_non_strict_check)
        
        # Orientation
        orient_layout = QHBoxLayout()
        orient_layout.addWidget(QLabel("Orientation Remap:"))
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(['none', 'mirror', 'rotate90'])
        orient_layout.addWidget(self.orientation_combo)
        orient_layout.addStretch()
        section.add_layout(orient_layout)
        
        return section
    
    def create_processing_section(self):
        """Create signal processing parameters section."""
        section = CollapsibleSection("⚙️ Signal Processing")
        
        # Paper style toggle
        self.paper_style_check = QCheckBox("Use paper-style preprocessing (120Hz HP, RMS only, 250ms windows)")
        self.paper_style_check.stateChanged.connect(self.on_paper_style_changed)
        section.add_widget(self.paper_style_check)
        
        # Window parameters
        window_layout = QGridLayout()
        window_layout.addWidget(QLabel("Window Size:"), 0, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(50, 1000)
        self.window_spin.setValue(200)
        self.window_spin.setSuffix(" ms")
        self.window_spin.valueChanged.connect(self.update_pipeline_display)
        window_layout.addWidget(self.window_spin, 0, 1)
        
        window_layout.addWidget(QLabel("Step Size:"), 1, 0)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(10, 500)
        self.step_spin.setValue(50)
        self.step_spin.setSuffix(" ms")
        self.step_spin.valueChanged.connect(self.update_pipeline_display)
        window_layout.addWidget(self.step_spin, 1, 1)
        
        # Overlap display
        self.overlap_label = QLabel()
        self.update_overlap_label()
        window_layout.addWidget(self.overlap_label, 2, 0, 1, 2)
        
        section.add_layout(window_layout)
        
        # Modality
        modality_layout = QHBoxLayout()
        modality_layout.addWidget(QLabel("Modality:"))
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(['emg', 'imu', 'both'])
        modality_layout.addWidget(self.modality_combo)
        modality_layout.addStretch()
        section.add_layout(modality_layout)
        
        # IMU settings (when modality includes IMU)
        imu_layout = QHBoxLayout()
        imu_layout.addWidget(QLabel("IMU Features:"))
        self.imu_features_combo = QComboBox()
        self.imu_features_combo.addItems(['mean', 'rich'])
        imu_layout.addWidget(self.imu_features_combo)
        imu_layout.addWidget(QLabel("Normalization:"))
        self.imu_norm_combo = QComboBox()
        self.imu_norm_combo.addItems(['zscore', 'robust'])
        imu_layout.addWidget(self.imu_norm_combo)
        imu_layout.addStretch()
        section.add_layout(imu_layout)
        
        return section
    
    def create_feature_section(self):
        """Create feature selection section."""
        section = CollapsibleSection("🔧 Feature Selection")
        
        desc = QLabel("Select features to extract from each EMG channel:")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #6B7280; font-size: 11px;")
        section.add_widget(desc)
        
        # Feature checkboxes
        self.feature_checks = {}
        for feature in AVAILABLE_FEATURES:
            check = QCheckBox(feature.replace('_', ' ').title())
            check.setChecked(True)
            check.stateChanged.connect(self.update_pipeline_display)
            self.feature_checks[feature] = check
            section.add_widget(check)
        
        # Select all/none buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self.set_all_features(True))
        btn_layout.addWidget(select_all_btn)
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(lambda: self.set_all_features(False))
        btn_layout.addWidget(select_none_btn)
        section.add_layout(btn_layout)
        
        return section
    
    def create_label_section(self):
        """Create label filtering section."""
        section = CollapsibleSection("🏷️ Label Filtering")
        
        # Ignore labels
        ignore_layout = QVBoxLayout()
        ignore_layout.addWidget(QLabel("Ignore these labels (comma-separated):"))
        self.ignore_labels_edit = QLineEdit()
        self.ignore_labels_edit.setText("Start, End, None, Unknown")
        self.ignore_labels_edit.textChanged.connect(self.update_pipeline_display)
        ignore_layout.addWidget(self.ignore_labels_edit)
        section.add_layout(ignore_layout)
        
        # Options
        self.ignore_case_check = QCheckBox("Case-insensitive label matching")
        self.ignore_case_check.setChecked(True)
        section.add_widget(self.ignore_case_check)
        
        self.keep_trial_check = QCheckBox("Keep trial numbers in labels (e.g., 'fist_3')")
        section.add_widget(self.keep_trial_check)
        
        return section
    
    def create_advanced_section(self):
        """Create advanced options section."""
        section = CollapsibleSection("🔬 Advanced Options")
        
        self.overwrite_check = QCheckBox("Overwrite existing output file")
        section.add_widget(self.overwrite_check)
        
        self.verbose_check = QCheckBox("Verbose logging")
        section.add_widget(self.verbose_check)
        
        # Config file
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Config File:"))
        self.config_file_edit = QLineEdit()
        self.config_file_edit.setPlaceholderText("Optional: load from config.json")
        config_layout.addWidget(self.config_file_edit)
        config_browse_btn = QPushButton("Browse...")
        config_browse_btn.clicked.connect(self.browse_config_file)
        config_layout.addWidget(config_browse_btn)
        section.add_layout(config_layout)
        
        return section
    
    def create_action_buttons(self):
        """Create action button layout."""
        layout = QHBoxLayout()
        
        # Left side buttons
        self.save_profile_btn = QPushButton("💾 Save Profile")
        self.save_profile_btn.clicked.connect(self.save_profile)
        layout.addWidget(self.save_profile_btn)
        
        self.load_profile_btn = QPushButton("📂 Load Profile")
        self.load_profile_btn.clicked.connect(self.load_profile)
        layout.addWidget(self.load_profile_btn)
        
        layout.addStretch()
        
        # Right side buttons
        self.build_btn = QPushButton("🚀 Build Dataset")
        self.build_btn.setProperty("accent", True)
        self.build_btn.clicked.connect(self.build_dataset)
        self.build_btn.setEnabled(False)
        layout.addWidget(self.build_btn)
        
        return layout
    
    def apply_styles(self):
        """Apply custom styles to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F6F7F9;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton[accent="true"] {
                background: #0A84FF;
                color: white;
                border: none;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton[accent="true"]:hover {
                background: #0077ED;
            }
            QPushButton[accent="true"]:disabled {
                background: #9CCBFF;
                color: #F3F8FF;
            }
        """)
    
    # Slots and event handlers
    def browse_root_dir(self):
        """Browse for root directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Project Root Directory"
        )
        if dir_path:
            self.root_dir_edit.setText(dir_path)
    
    def browse_config_file(self):
        """Browse for config file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config File", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.config_file_edit.setText(file_path)
            self.load_config_from_file(file_path)
    
    def on_root_dir_changed(self, text):
        """Handle root directory change."""
        self.root_dir = text if text else None
        self.build_btn.setEnabled(bool(self.root_dir))
        if self.root_dir and os.path.isdir(self.root_dir):
            self.discover_files()
    
    def on_file_type_changed(self, file_type):
        """Handle file type change."""
        self.discover_files()
        self.update_pipeline_display()
    
    def on_paper_style_changed(self, state):
        """Handle paper style toggle."""
        if state == Qt.Checked:
            self.window_spin.setValue(250)
            self.step_spin.setValue(250)
            # Disable window/step editing in paper mode
            self.window_spin.setEnabled(False)
            self.step_spin.setEnabled(False)
        else:
            self.window_spin.setEnabled(True)
            self.step_spin.setEnabled(True)
        self.update_pipeline_display()
    
    def update_overlap_label(self):
        """Update the overlap percentage label."""
        window = self.window_spin.value()
        step = self.step_spin.value()
        if window > 0:
            overlap = int((1 - step/window) * 100)
            self.overlap_label.setText(f"Overlap: {overlap}%")
            self.overlap_label.setStyleSheet("color: #6B7280; font-style: italic;")
    
    def discover_files(self):
        """Discover files in the root directory."""
        if not self.root_dir or not os.path.isdir(self.root_dir):
            return
        
        file_type = self.file_type_combo.currentText()
        ext = f".{file_type}"
        
        # Find data files (search in 'raw' subdirectory if it exists, otherwise search everywhere)
        self.discovered_files = []
        self.file_list.clear()
        
        raw_dir = os.path.join(self.root_dir, 'raw')
        search_dirs = [raw_dir] if os.path.isdir(raw_dir) else [self.root_dir]
        
        for search_dir in search_dirs:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(ext):
                        full_path = os.path.join(root, file)
                        self.discovered_files.append(full_path)
                        rel_path = os.path.relpath(full_path, self.root_dir)
                        self.file_list.addItem(rel_path)
        
        # Update data count label
        self.data_count_label.setText(
            f"Found {len(self.discovered_files)} {file_type.upper()} file(s)"
        )
        if len(self.discovered_files) == 0:
            self.data_count_label.setStyleSheet("color: #DC2626; font-weight: bold; font-size: 11px;")
        else:
            self.data_count_label.setStyleSheet("color: #0369A1; font-weight: bold; font-size: 11px;")
        
        # Find event files (search in 'events' subdirectory if it exists, otherwise search everywhere)
        self.event_files = []
        self.event_list.clear()
        
        events_dir = os.path.join(self.root_dir, 'events')
        event_search_dirs = [events_dir] if os.path.isdir(events_dir) else [self.root_dir]
        
        for search_dir in event_search_dirs:
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    # Look for various event file patterns
                    if (file.endswith('_events.txt') or 
                        file.endswith('.events') or
                        file.endswith('_event.txt') or
                        file.endswith('.event') or
                        (file.endswith('.txt') and 'event' in file.lower())):
                        full_path = os.path.join(root, file)
                        self.event_files.append(full_path)
                        rel_path = os.path.relpath(full_path, self.root_dir)
                        self.event_list.addItem(rel_path)
        
        # Update event count label
        self.event_count_label.setText(
            f"Found {len(self.event_files)} event file(s)"
        )
        if len(self.event_files) == 0:
            self.event_count_label.setStyleSheet("color: #DC2626; font-weight: bold; font-size: 11px;")
        else:
            self.event_count_label.setStyleSheet("color: #92400E; font-weight: bold; font-size: 11px;")
        
        # Update status bar
        status_msg = f"📊 {len(self.discovered_files)} data file(s) | 🏷️ {len(self.event_files)} event file(s)"
        self.statusBar().showMessage(status_msg)
    
    def update_pipeline_display(self):
        """Update the pipeline visualization with current settings."""
        self.update_overlap_label()
        
        params = self.get_current_params()
        self.pipeline_widget.update_pipeline(params)
    
    def set_all_features(self, checked):
        """Set all feature checkboxes to checked/unchecked."""
        for check in self.feature_checks.values():
            check.setChecked(checked)
    
    def get_current_params(self) -> Dict:
        """Get current parameter settings as a dictionary."""
        # Get selected features
        selected_features = [
            name for name, check in self.feature_checks.items() 
            if check.isChecked()
        ]
        
        # Get ignore labels
        ignore_labels_text = self.ignore_labels_edit.text().strip()
        ignore_labels = [l.strip() for l in ignore_labels_text.split(',') if l.strip()]
        
        # Get channel map or channels
        channel_map = self.channel_map_combo.currentText()
        if channel_map == 'None':
            channel_map = None
        
        channels = self.channels_edit.text().strip() or None
        
        params = {
            'root_dir': self.root_dir,
            'file_type': self.file_type_combo.currentText(),
            'multi_file': self.multi_file_check.isChecked(),
            'label': self.label_edit.text().strip(),
            'window_ms': self.window_spin.value(),
            'step_ms': self.step_spin.value(),
            'paper_style': self.paper_style_check.isChecked(),
            'channels': channels,
            'channel_map': channel_map,
            'mapping_non_strict': self.mapping_non_strict_check.isChecked(),
            'orientation_remap': self.orientation_combo.currentText(),
            'modality': self.modality_combo.currentText(),
            'imu_features': self.imu_features_combo.currentText(),
            'imu_norm': self.imu_norm_combo.currentText(),
            'features': selected_features,
            'ignore_labels': ignore_labels,
            'ignore_case': self.ignore_case_check.isChecked(),
            'keep_trial_label': self.keep_trial_check.isChecked(),
            'merge_pattern': self.merge_pattern_edit.text().strip() or None,
            'exclude_pattern': self.exclude_pattern_edit.text().strip() or None,
            'overwrite': self.overwrite_check.isChecked(),
            'verbose': self.verbose_check.isChecked(),
        }
        
        return params
    
    def save_profile(self):
        """Save current configuration to a profile file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", "", "JSON Files (*.json)"
        )
        if file_path:
            params = self.get_current_params()
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=2)
            self.statusBar().showMessage(f"Profile saved: {file_path}")
            QMessageBox.information(self, "Success", "Profile saved successfully!")
    
    def load_profile(self):
        """Load configuration from a profile file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Profile", "", "JSON Files (*.json)"
        )
        if file_path:
            self.load_config_from_file(file_path)
    
    def load_config_from_file(self, file_path):
        """Load configuration from JSON file."""
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
            
            # Update UI with loaded parameters
            if 'root_dir' in params:
                self.root_dir_edit.setText(params['root_dir'])
            if 'file_type' in params:
                self.file_type_combo.setCurrentText(params['file_type'])
            if 'multi_file' in params:
                self.multi_file_check.setChecked(params['multi_file'])
            if 'label' in params:
                self.label_edit.setText(params['label'])
            if 'window_ms' in params:
                self.window_spin.setValue(params['window_ms'])
            if 'step_ms' in params:
                self.step_spin.setValue(params['step_ms'])
            if 'paper_style' in params:
                self.paper_style_check.setChecked(params['paper_style'])
            if 'channels' in params and params['channels']:
                self.channels_edit.setText(params['channels'])
            if 'channel_map' in params and params['channel_map']:
                self.channel_map_combo.setCurrentText(params['channel_map'])
            if 'modality' in params:
                self.modality_combo.setCurrentText(params['modality'])
            if 'features' in params:
                for name, check in self.feature_checks.items():
                    check.setChecked(name in params['features'])
            if 'ignore_labels' in params:
                self.ignore_labels_edit.setText(', '.join(params['ignore_labels']))
            
            self.update_pipeline_display()
            self.statusBar().showMessage(f"Profile loaded: {file_path}")
            QMessageBox.information(self, "Success", "Profile loaded successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load profile:\n{str(e)}")
    
    def build_dataset(self):
        """Build the dataset with current configuration."""
        params = self.get_current_params()
        
        # Validate inputs
        if not params['root_dir'] or not os.path.isdir(params['root_dir']):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid root directory.")
            return
        
        if not params['features']:
            QMessageBox.warning(self, "Invalid Input", "Please select at least one feature.")
            return
        
        # Build command
        script_path = Path(__file__).parent.parent / "gesture_classifier" / "1_build_dataset.py"
        
        cmd_parts = [sys.executable, str(script_path)]
        cmd_parts.extend(['--root_dir', params['root_dir']])
        cmd_parts.extend(['--file_type', params['file_type']])
        
        if params['multi_file']:
            cmd_parts.append('--multi_file')
        
        if params['label']:
            cmd_parts.extend(['--label', params['label']])
        
        cmd_parts.extend(['--window_ms', str(params['window_ms'])])
        cmd_parts.extend(['--step_ms', str(params['step_ms'])])
        
        if params['paper_style']:
            cmd_parts.append('--paper_style')
        
        if params['channels']:
            cmd_parts.extend(['--channels', params['channels']])
        
        if params['channel_map']:
            cmd_parts.extend(['--channel_map', params['channel_map']])
        
        if params['mapping_non_strict']:
            cmd_parts.append('--mapping_non_strict')
        
        cmd_parts.extend(['--orientation_remap', params['orientation_remap']])
        cmd_parts.extend(['--modality', params['modality']])
        cmd_parts.extend(['--imu_features', params['imu_features']])
        cmd_parts.extend(['--imu_norm', params['imu_norm']])
        
        if params['ignore_labels']:
            cmd_parts.extend(['--ignore_labels'] + params['ignore_labels'])
        
        if params['ignore_case']:
            cmd_parts.append('--ignore_case')
        
        if params['keep_trial_label']:
            cmd_parts.append('--keep_trial_label')
        
        if params['merge_pattern']:
            cmd_parts.extend(['--merge_pattern', params['merge_pattern']])
        
        if params['exclude_pattern']:
            cmd_parts.extend(['--exclude_pattern', params['exclude_pattern']])
        
        if params['overwrite']:
            cmd_parts.append('--overwrite')
        
        if params['verbose']:
            cmd_parts.append('--verbose')
        
        # Show command for user
        cmd_str = ' '.join(cmd_parts)
        msg = QMessageBox()
        msg.setWindowTitle("Build Dataset")
        msg.setText("Ready to build dataset with the following command:")
        msg.setDetailedText(cmd_str)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)
        
        if msg.exec_() == QMessageBox.Ok:
            # Execute command
            import subprocess
            try:
                self.statusBar().showMessage("Building dataset...")
                self.build_btn.setEnabled(False)
                
                # Run in terminal/console
                if sys.platform == 'win32':
                    subprocess.Popen(['start', 'cmd', '/k'] + cmd_parts, shell=True)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', '-a', 'Terminal'] + cmd_parts)
                else:
                    subprocess.Popen(['x-terminal-emulator', '-e'] + cmd_parts)
                
                self.statusBar().showMessage("Dataset building started in external terminal")
                QMessageBox.information(
                    self, "Success", 
                    "Dataset building started!\nCheck the terminal window for progress."
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start build:\n{str(e)}")
            finally:
                self.build_btn.setEnabled(True)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Apply light theme
    palette = QtGui.QPalette()
    palette.setColor(palette.Window, QtGui.QColor("#F6F7F9"))
    palette.setColor(palette.Base, QtGui.QColor("#FFFFFF"))
    palette.setColor(palette.Text, QtGui.QColor("#1C1C1E"))
    palette.setColor(palette.Button, QtGui.QColor("#FFFFFF"))
    palette.setColor(palette.ButtonText, QtGui.QColor("#1C1C1E"))
    palette.setColor(palette.Highlight, QtGui.QColor("#0A84FF"))
    palette.setColor(palette.HighlightedText, QtGui.QColor("#FFFFFF"))
    app.setPalette(palette)
    
    window = DatasetBuilderGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
