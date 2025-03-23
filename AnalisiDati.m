% Directory contenente i file CSV
dataDir = './Data_Analysis/Tickers_file';

% Lista dei file CSV di interesse
fileList = { ...
    'XLE_data.csv', ...
    'AAPL_data.csv', ...
    'AMZN_data.csv', ...
    'XLY_data.csv',... 
    'XLI_data.csv',...
    'ITA_data.csv', ...
    'NFLX_data.csv', ...
    'NKE_data.csv', ...
    'NVDA_data.csv', ...
    'XLE_data.csv', ...
    'WMT_data.csv', ...
    'XLV_data.csv', ...
    'XLF_data.csv', ...
    'XLK_data.csv', ...
    'XOM_data.csv'};

% Leggi i file presenti nella directory
allFiles = dir(fullfile(dataDir, '*.csv'));
allFilenames = {allFiles.name};

% Filtra i file che sono nella lista di interesse
filteredFilenames = intersect(fileList, allFilenames);

% Controllo se alcuni file della lista non sono presenti
missingFiles = setdiff(fileList, filteredFilenames);
if ~isempty(missingFiles)
    fprintf('I seguenti file non sono stati trovati nella directory "%s":\n', dataDir);
    disp(missingFiles);
end

% Itera solo sui file filtrati
for i = 1:length(filteredFilenames)
    % Percorso completo del file
    filepath = fullfile(dataDir, filteredFilenames{i});
    
    % Leggi la tabella dal file
    data = readtable(filepath);

    % Controllo dell'esistenza di tutte le colonne richieste
    requiredVars = {'xReturn', 'AdjClose', 'Open', 'Volume'};
    for rv = requiredVars
        if ~ismember(rv{1}, data.Properties.VariableNames)
            error('La colonna "%s" non è presente in %s', rv{1}, filteredFilenames{i});
        end
    end
    
    % Estrai i dati dalle colonne di interesse
    returns     = data.xReturn;
    adjcloses  = data.AdjClose;
    opens      = data.Open;
    volumes    = data.Volume;
    
    % Ricavo il nome del file senza estensione
    [~, name, ~] = fileparts(filteredFilenames{i});
    % Sostituisco eventuali punti ('.') con underscore ('_')
    name = strrep(name, '.', '_');
    
    % Creo i nomi delle variabili dinamiche
    returnVarName   = ['return_'     name];
    closeVarName   = ['AdjClose_'  name];
    openVarName    = ['Open_'      name];
    volumeVarName  = ['Volume_'    name];
    
    % Assegno le variabili nel workspace "base"
    assignin('base', returnVarName,  returns);
    assignin('base', closeVarName,  adjcloses);
    assignin('base', openVarName,   opens);
    assignin('base', volumeVarName, volumes);
end

disp('I dati sono stati caricati con successo nel workspace.');
