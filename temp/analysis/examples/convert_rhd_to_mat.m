% This example script opens a .rhd file and converts it to a .mat file preserving all metadata and info. 
% Usage:
%   1.  Requires the +io package as a dependency which can be pulled into the file directory from a terminal:
%       ```
%       git clone https://github.com/Neuro-Mechatronics-Interfaces/matlab_package__io.git +io
%       ```
%   2. Open MATLAB and run "io.setup" in the Command Window to install other required repos.

clear; clc;
% ======================================================================================
% ========== The only portion that needs to be adjusted by user for new files ==========

% If you know the number of devices used, you can define the labels to give
% them here, usually the muscles the grids are located over
CHANNEL_LABEL = [
  "1", "TA";        % Can assign labels to individual channels, or...
  "2:64", "TA";     % ...assign to range of values as string
  "65:128", "TA";  % Just make sure all channels have a label
  % "129:192", "SOL";
  % "193:256", "NG";
  % You can add/remove channels and labels as needed
];


%fname = "C:/example/path/to/IsometricFlexionRamp_1_241022_150834/IsometricFlexionRamp_241022_150834.rhd";
%x = io.read_Intan_RHD2000_file(fname);              % Define full file path and pass in to the reader...
%x = io.read_Intan_RHD2000_file();                   % ... or call up the GUI to select the file.
x = io.read_Intan_RHD2000_files("Concatenate",true); % Load multiple fies with the GUI as well

%% ======================================================================================

% Let's parse out the data
% Get sample rate from unipolar EMG channels
sample_rate = x.frequency_parameters.amplifier_sample_rate;

% We can apply a 10Hz high-pass filter to the HDEMG channel data
[b,a] = butter(3,10/(sample_rate/2),'high');
uni = filtfilt(b,a,x.amplifier_data')';

% Arrangement of channels is important so the order needs to be reordered.
% Based on 128CH-HDEMG Wrist Sleeve channel layout, 8x8 order of channels 1:64 need
% to be transposed. Same transpose applied to 65:128. This repeats for every
% 128 channels if using more than one device...
%uni(1:64,:) = reshape(gradient(reshape(uni(1:64,:),8,8,[])),64,[]);     
%uni(65:128,:) = reshape(gradient(reshape(uni(65:128,:),8,8,[])),64,[]);

% Get the time data in seconds
t_s = x.t_amplifier; % Time vector

% Auxiliary input data, choosing combine to one channel, or keep separate
aux_data = x.aux_input_data;
%aux_data = rms(aux(1:3,:)-median(x.aux(1:3,:),2),1);

% The analog data
analog = x.board_adc_data;


% The auxiliary input for the RHD chips is actually 3 channels with a
% separate sampling rate, need to interpolate to match it with the data
fs_aux = x.frequency_parameters.aux_input_sample_rate;

% Get the size of aux
[num_channels_aux, num_samples_aux] = size(aux_data);  % Ensure correct dimension reference
aux_resampled = zeros(num_channels_aux, round((sample_rate / fs_aux) * num_samples_aux));

% Apply filtering to each channel separately and interpolate data to match fs
[b, a] = butter(3, 1 / (fs_aux / 2), 'low');
for ch = 1:num_channels_aux
    filtered_signal = filtfilt(b, a, aux_data(ch, :));
    aux_resampled(ch, :) = interp1(1:num_samples_aux, filtered_signal, linspace(1, num_samples_aux, size(aux_resampled, 2)));
end
aux = aux_resampled;

% Any syncronization or TTL signal inputs
%sync = -x.board_adc_data; % Any sync signals captured

% Custom channel mapping based on 128CH-HDEMG Wrist Sleeve channel layout
grid1 = flip(reshape(1:64,8,8)');
grid2 = flip(reshape(65:128,8,8)');
single_device_mapping = [grid2, grid1];
%disp(single_device_mapping)

% Obtain info about the data and device itself
[N_channels, N_samples] = size(uni); % Number of channels and samples in file
channels_per_device = 128; % Fixed number of channels per device
num_devices = N_channels / channels_per_device; % Total devices

% Pre-allocate cell array to store channel mappings and labels for each device
channel_mappings = cell(1, num_devices);  
channel_labels = cell(1, num_devices);   
for i = 1:num_devices
    channel_mappings{i} = single_device_mapping + (i-1) * channels_per_device;
    channel_labels{i} = strings(8,16);  % Initialize empty label grid
end

% Create a full array of channel labels (1xN_channels)
channel_array(1, :) = string(1:N_channels);

% Assign labels based on user input
for i = 1:size(CHANNEL_LABEL, 1)
    range_val = CHANNEL_LABEL(i, 1); % Get the range
    label = CHANNEL_LABEL(i, 2);     % Get the corresponding label
    
    % Convert range to numeric indices
    if ischar(range_val) || isstring(range_val)  % If it's a string, evaluate it
        range_indices = eval(range_val); 
    elseif isnumeric(range_val)  % If it's already numeric, use it directly
        range_indices = range_val; 
    end
    
    % Ensure indices do not exceed N_channels and assign label to valid indices
    range_indices = range_indices(range_indices <= N_channels);    
    channel_array(2, range_indices) = label;
end

% Convert labels into 8x16 grid per device
for dev = 1:num_devices
    for row = 1:8
        for col = 1:16
            ch = channel_mappings{dev}(row, col); % Get the actual channel number
            if ch <= N_channels  % Ensure it's a valid channel
                channel_labels{dev}(row, col) = channel_array(2, ch);
            end
        end
    end
end

% Display the formatted channel labels for verification
for dev = 1:num_devices
    fprintf('Device %d Channel Labels:\n', dev);
    disp(channel_labels{dev});
end

% Halve the precision accuraccy to reduce file size when exporting
uni = single(uni);
aux = single(aux);
analog = single(analog);
x = rmfield(x, 'amplifier_data'); % Remove if using 'uni' instead

% Export data to .mat file
export_dir = fullfile(sprintf('%s', x.path));
mat_file = sprintf('%s.mat', x.filename);
sprintf("Saving file to %s", fullfile(export_dir, mat_file))
save(fullfile(export_dir, mat_file),...
    'x',...
    'uni','sample_rate','N_channels','N_samples','aux','analog','channel_mappings','channel_labels',...
    '-v7'... % Change to v7.3 if issues with exporting
    );
disp("Export complete")