function export_essex_native_tables(source_root, export_root)
% EXPORT_ESSEX_NATIVE_TABLES Convert original Essex MATLAB table objects.
% Presented by: Reza Saadatyar (2026)
%
% Requires MATLAB Online/Desktop; not executable in ordinary Python Colab.
% Source: Figshare article 30225739, version 13 (original MATLAB.zip).
% Usage:
%   export_essex_native_tables('/path/to/unzipped', '/path/to/NEW/export')
%
% This preserves source identities and MATLAB column labels, without
% interpolating values, assigning undocumented units, or training models.
% Keep the resulting numerical archive outside the public Git repository.

if nargin ~= 2, error('Specify source_root and a NEW export_root.'); end
source_root = char(source_root);
export_root = char(export_root);
if ~isfolder(source_root), error('Source directory not found.'); end
if exist(export_root, 'dir') || exist(export_root, 'file')
    error('Export destination already exists; choose a new empty path.');
end

files = dir(fullfile(source_root, '**', 'subj*.mat'));
files = files(~[files.isdir]);
if numel(files) ~= 40
    error('Expected exactly 40 original subject MAT files, found %d.',numel(files));
end
if numel(unique(lower(string({files.name})))) ~= 40
    error('Source subject filenames are not unique.');
end
mkdir(export_root);
mkdir(fullfile(export_root,'numeric'));

rows = repmat(empty_row(),0,1);
n_records = 0;
for file_index = 1:numel(files)
    filename = files(file_index).name;
    subject = lower(erase(string(filename),'.mat'));
    if isempty(regexp(char(subject),'^subj\d+$','once'))
        error('Invalid source subject filename: %s',filename);
    end
    filedata = load(fullfile(files(file_index).folder,filename),'Data');
    if ~isfield(filedata,'Data') || ~isstruct(filedata.Data)
        error('Missing original Data struct in %s',filename);
    end
    records = filedata.Data(:);
    n_records = n_records + numel(records);
    for recindex = 1:numel(records)
        record = records(recindex);
        required = {'SubjID','Task','UpdatedTrialID','IMU','e1e3'};
        if ~all(isfield(record,required))
            error('Required fields missing in %s, record %d',filename,recindex);
        end
        if lower(strtrim(string(record.SubjID))) ~= subject
            error('Source subject does not match file: %s',filename);
        end
        task = strtrim(string(record.Task));
        trial = strtrim(string(record.UpdatedTrialID));
        if strlength(task) == 0 || strlength(trial) == 0
            error('Task or trial ID is missing for %s',filename);
        end
        matlab_fields = {'IMU','e1e3'};
        modality_names = {'IMU','FE'};
        for m = 1:2
            modality = string(modality_names{m});
            raw = record.(matlab_fields{m});
            entry = empty_row();
            entry.subject_id = subject;
            entry.source_file = string(filename);
            entry.source_record_index = recindex - 1;
            entry.task = task;
            entry.updated_trial_id = trial;
            entry.modality = modality;
            entry.source_class = string(class(raw));
            [values, column_names, result] = recover_numeric(raw);
            entry.status = result;
            if result == "exported"
                entry.n_time = size(values,1);
                entry.n_channels = size(values,2);
                entry.column_names = jsonencode(cellstr(column_names));
                outname = sprintf('%s__r%04d__%s.mat', ...
                    char(subject),recindex-1,char(modality));
                entry.export_relpath = string(['numeric/' outname]);
                outpath = fullfile(export_root,'numeric',outname);
                if exist(outpath,'file'), error('Duplicate export: %s',outname); end
                save(outpath,'values','column_names','-v7');
            end
            rows(end+1,1) = entry; %#ok<AGROW>
        end
    end
end

if numel(rows) ~= 2*n_records
    error('Export manifest record accounting failed.');
end
manifest = struct2table(rows);
writetable(manifest,fullfile(export_root,'export_manifest.csv'));
fid = fopen(fullfile(export_root,'export_readme.txt'),'w');
fprintf(fid,'Essex Figshare 30225739 v13; original MATLAB-native table conversion.\n');
fprintf(fid,'Numerical payloads remain unverified for units/FE time alignment.\n');
fprintf(fid,'Keep exported numeric MAT files outside the public Git repository.\n');
fclose(fid);
archive = fullfile(fileparts(export_root),'native_essex_export.zip');
if exist(archive,'file'), error('Output ZIP already exists: %s',archive); end
zip(archive,{'export_manifest.csv','export_readme.txt','numeric'},export_root);
fprintf('Original records: %d; export manifest rows: %d\n',n_records,height(manifest));
fprintf('Numeric payloads exported: %d\n',nnz(manifest.status=="exported"));
fprintf('Upload to Colab as /content/native_essex_export.zip: %s\n',archive);
end

function row = empty_row()
row = struct('subject_id',"",'source_file',"", ...
    'source_record_index',0,'task',"",'updated_trial_id',"", ...
    'modality',"",'source_class',"",'status',"", ...
    'n_time',0,'n_channels',0,'column_names',"",'export_relpath',"");
end

function [values, names, status] = recover_numeric(raw)
% Only native numeric scalar-column tables or 2-D numeric matrices.
% No silent conversion of complex, nonfinite, cell, or unknown MATLAB objects.
values = []; names = strings(1,0); status = "unsupported_class";
if isempty(raw), status = "empty"; return; end
if istable(raw) || istimetable(raw)
    names = string(raw.Properties.VariableNames);
    for j = 1:width(raw)
        col = raw{:,j};
        if ~isnumeric(col) || ~isreal(col) || size(col,2) ~= 1
            status = "unsupported_column"; return;
        end
    end
    values = table2array(raw);
elseif isnumeric(raw) && ismatrix(raw)
    values = raw;
    names = "UNLABELLED_" + string(1:size(raw,2));
else
    return;
end
if isempty(values)
    status = "empty";
elseif ~isreal(values) || ~all(isfinite(values(:)))
    status = "nonfinite";
elseif numel(unique(names)) ~= numel(names)
    status = "duplicate_columns";
else
    values = double(values);
    status = "exported";
end
end
