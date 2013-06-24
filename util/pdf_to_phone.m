function [pdf_to_phone, num_to_txt] = pdf_to_phone()

%% Build map of pdf ids to phone ids.  Returns the vector of 
%% Phone Ids logically indexed by pdf id.  Also returns a phone id
%% to phone string mapping in num_to_txt

% vars
%num_pdfs = 7711;
num_pdfs = 3034;
num_phones = 46;
pdf_to_phone_cell = cell(num_pdfs,1);
pdf_to_phone = zeros(num_pdfs,1);
num_to_txt = cell(num_phones,1);

% start by generating map of pdf id to all possible phones from file
% pdf-to-phone.txt
%fid = fopen('pdf-phone-map-big.txt');
fid = fopen('pdf-phone-map.txt');

index=1;

while 1
    nline = fgetl(fid);
    if ~ischar(nline), break, end;
    if nline=='-'
        index=index+1;
    else
        pdf_to_phone_cell{index} = [pdf_to_phone_cell{index}, str2num(nline)];
    end;

end;


fclose(fid);

% pick base phone per group from sets.int to tie
% all the phones of that group to
sets = textread('sets.int');
for i=1:num_pdfs
    list=pdf_to_phone_cell{i};
    
    % find which set the first phone in each pdf list is and assign
    % the pdf to the first phone of that set
    [m n] = find(sets==list(1));
    pdf_to_phone(i) = m;
end;
% at this point every pdf should go to a single phone


% Also get integer to phone text mapping
fid = fopen('base_phones.txt');
for i=1:num_phones
    num_to_txt{i} = fgetl(fid);
end;
fclose(fid);

end
