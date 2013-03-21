
%%First we build map

%start by generating map of pdf id to all possible phones from file
%pdf-to-phone.txt
num_pdfs = 3034;
num_phones = 46;
pdf_to_phone = cell(num_pdfs,1);

fid = fopen('pdf-phone-map.txt');

index=1;

while 1
    nline = fgetl(fid);
    if ~ischar(nline), break, end;
    if nline=='-'
        index=index+1;
    else
        pdf_to_phone{index} = [pdf_to_phone{index}, str2num(nline)];
    end

end


%perhaps consider picking one phone per group from sets.int to tie
%all the phones of that group to
sets = textread('sets.int');
for i=1:num_pdfs
    list=pdf_to_phone{i};
    
    %find which set the first phone in each pdf list is and assign
    %the pdf to the first phone of that set
    [m n] = find(sets==list(1));
    pdf_to_phone{i} = [m];
end
%at this point every pdf should go to a single phone

%%Now we compare NN alignments to forced alignment from base model

%Need NN Model and data
nn_model = '../simple-hybrid/2hidden_2048_relu_fmllr/spNet_5';
dat_dir = '/afs/cs.stanford.edu/u/awni/luster_awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5/exp/nn_data_dev/';
addpath ../simple-hybrid;

%load network params
load(nn_model,'eI','theta');
eI.useGpu = 0; %don't use the gpu as likely too much memory

disp(eI);

%load data
%Load features and data (keys and utt sizes)
fn=1; %set file number
[feats utt_dat alis] = load_kaldi_data(dat_dir,fn,eI.inputDim);
assert(size(feats,1)==sum(utt_dat.sizes));

%choose an utterance to view
uttnum=1;
fStart = sum(utt_dat.sizes(1:uttnum-1))+1;
fEnd = fStart+utt_dat.sizes(uttnum)-1;
uttFeats = feats(fStart:fEnd,:);
uttAlis = alis(fStart:fEnd);

%forward prop utterance
[c, g, nC, nE, ceC, wC, output] = spNetCostSlave(theta,eI,uttFeats',[],1);

%take index of max probability for each frame to generate alignment
%vector
[p uttAlisNN] = max(output);
uttAlisNN=uttAlisNN';

%map both from pdf id to phone
num_frames = length(uttAlis);
uttPhonesNN = zeros(num_phones,num_frames);
uttPhones = zeros(num_phones,num_frames);
for a=1:num_frames
    uttPhonesNN(pdf_to_phone{uttAlisNN(a)},a) = 1;
    uttPhones(pdf_to_phone{uttAlis(a)},a) =  -1;
end;



%compare
diff=uttPhonesNN+uttPhones;
same=2*uttPhonesNN+uttPhones;
same=0.5*(same==1);
imagesc(diff(:,1:50)+same(:,1:50));
hold on;
xlabel('Frames of utterance');
ylabel('Phone Label 1-46');
colorbar;



