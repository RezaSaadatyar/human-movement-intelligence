function inspect_essex_matlab_native(subjects)
% Inspect actual MATLAB field classes and table schemas without exporting values.
% Default: subj04 and subj16 only. Raw MAT files are never modified.
if nargin < 1 || isempty(subjects)
    subjects = {'subj04','subj16'};
elseif ischar(subjects) || (isstring(subjects) && isscalar(subjects))
    subjects = cellstr(subjects);
elseif isstring(subjects)
    subjects = cellstr(subjects);
end
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
dataDir = fullfile(repoRoot,'data','raw','essex_femoral_neck_loading', ...
                   'MATLAB','Processed Data');
if numel(subjects)==1 && strcmp(subjects{1},'all')
    listing = dir(fullfile(dataDir,'subj*.mat'));
    subjects = cellfun(@(n) erase(n,'.mat'),{listing.name},'UniformOutput',false);
end
fields = {'IMU','ID','FP','JCF','MuscleF','e1e3','Cycle','TrialNumUpdated'};
reports = cell(numel(subjects),1);
for s=1:numel(subjects)
    sid=char(subjects{s});
    if isempty(regexp(sid,'^subj[0-9]+$','once'))
        error('Invalid subject identifier: %s',sid);
    end
    path=fullfile(dataDir,[sid '.mat']);
    if ~isfile(path)
        error('File not found: %s',path);
    end
    source=load(path,'Data');
    if ~isfield(source,'Data')
        error('Data absent: %s',path);
    end
    Data=source.Data;
    n=numel(Data);
    fieldReports=cell(numel(fields),1);
    for f=1:numel(fields)
        key=fields{f};
        summary=struct('name',key,'missing',0,'empty',0,'nonempty',0, ...
                       'class_names',{{}},'class_counts',[], ...
                       'first_nonempty_metadata',[]);
        seenClasses={};
        for k=1:n
            if iscell(Data)
                record=Data{k};
            elseif isstruct(Data)
                record=Data(k);
            else
                error('Unsupported Data type: %s',class(Data));
            end
            if ~isstruct(record) || ~isfield(record,key)
                summary.missing=summary.missing+1;
                continue;
            end
            value=record.(key);
            seenClasses{end+1}=class(value); %#ok<AGROW>
            if isempty(value)
                summary.empty=summary.empty+1;
            else
                summary.nonempty=summary.nonempty+1;
                if isempty(summary.first_nonempty_metadata)
                    summary.first_nonempty_metadata=describe_value(value);
                end
            end
        end
        classes=unique(seenClasses,'stable');
        counts=zeros(1,numel(classes));
        for j=1:numel(classes)
            counts(j)=sum(strcmp(seenClasses,classes{j}));
        end
        summary.class_names=classes;
        summary.class_counts=counts;
        fieldReports{f}=summary;
    end
    item=struct();
    item.file=[sid '.mat'];
    item.Data_class=class(Data);
    item.Data_size=size(Data);
    item.record_count=n;
    item.field_audits=fieldReports;
    reports{s}=item;
    e=fieldReports{strcmp(fields,'e1e3')};
    fprintf('%s: %d records | e1e3 empty=%d nonempty=%d\n', ...
            sid,n,e.empty,e.nonempty);
    if ~isempty(e.first_nonempty_metadata)
        fprintf('  First populated e1e3 MATLAB class: %s\n', ...
                e.first_nonempty_metadata.class);
    end
end
report=struct('source','Figshare 30225739 v13', ...
    'scope','MATLAB-native metadata and shapes; no signal values', ...
    'subjects',{reports});
out=fullfile(repoRoot,'outputs','essex_matlab_native_schema.json');
outDir=fileparts(out);
if ~exist(outDir,'dir')
    mkdir(outDir);
end
fid=fopen(out,'w');
if fid<0
    error('Cannot open output report: %s',out);
end
closeFile=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(report));
fprintf('Report saved: %s\n',out);
end

function description=describe_value(value)
% Record names, shapes and classes; no sensor values or participant notes.
description=struct();
description.class=class(value);
description.size=size(value);
description.is_table=istable(value);
description.column_names={};
description.column_classes={};
description.column_shapes={};
description.struct_field_names={};
if istable(value)
    names=value.Properties.VariableNames;
    description.column_names=names;
    classes=cell(1,numel(names));
    shapes=cell(1,numel(names));
    for j=1:numel(names)
        column=value.(names{j});
        classes{j}=class(column);
        shapes{j}=size(column);
    end
    description.column_classes=classes;
    description.column_shapes=shapes;
elseif isstruct(value)
    description.struct_field_names=fieldnames(value);
end
end
