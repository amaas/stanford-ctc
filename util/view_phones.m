%% Config

%Need NN Model and data
nn_model = '/afs/cs/u/amaas/scratch/runtime/audio/phoneme_class/icml2013/single_2048_2048_2048_2048_relu/spNet_restart_e1';
nn_model = '/afs/cs.stanford.edu/u/awni/luster_awni/kaldi-stanford/stanford-nnet/simple-hybrid/4hidden_2048_relu_fbank_nomomentum/spNet_0';
dat_dir = '/afs/cs.stanford.edu/u/awni/luster_awni/kaldi-stanford/kaldi-trunk/egs/swbd/s5/exp/nn_data_full_fbank_pca_512/';
addpath ../simple-hybrid;


%% First we build map

%start by getting map of pdf id to all possible phones
num_pdfs = 3034;
num_phones = 46;

[pdf_phone_map num_to_txt] = pdf_to_phone();

%% Now we compare NN alignments to forced alignment from base model

% load network params
load(nn_model,'eI','theta');
eI.useGpu = 0; % don't use the gpu as likely too much memory

disp(eI);

% load data
% Load features and data (keys and utt sizes)
fn=1; % set file number
[feats utt_dat alis] = load_kaldi_data(dat_dir,fn,eI.inputDim);
assert(size(feats,1)==sum(utt_dat.sizes));

% get only first 1000 utts
%utt_dat.keys = utt_dat.keys(1:1000);
%utt_dat.sizes = utt_dat.sizes(1:1000);
%feats = feats(1:sum(utt_dat.sizes),:);
%alis = alis(1:sum(utt_dat.sizes));
%feats = feats(1:100000,:);
%alis = alis(1:100000);



% forward prop all utts for phone accuracy calculation and for
% building confusion matrix
numPhonesRight=0;
numSonemesRight=0;
chunkSize = 100; %Size of utterance chunks to forward prop at a
                 %time
numChunks = ceil(length(utt_dat.keys)/chunkSize);
numFramesDone = 0; %Number of total frames written
confusion = zeros(num_phones);
for i=1:numChunks

    %Get subset of keys and subset of sizes to forward prop and
    %write
    if i==numChunks
        subKeys=utt_dat.keys((i-1)*chunkSize+1:end);
        subSizes=utt_dat.sizes((i-1)*chunkSize+1:end,:);
    else
        subKeys=utt_dat.keys((i-1)*chunkSize+1:i*chunkSize);
        subSizes=utt_dat.sizes((i-1)*chunkSize+1:i*chunkSize);
    end
    input = feats(numFramesDone+1:numFramesDone+sum(subSizes),:);
    uttAlis = alis(numFramesDone+1:numFramesDone+sum(subSizes));
    %input = feats;
    %uttAlis = alis;
    [c, g, nC, nE, ceC, wC, output] = spNetCostSlave(theta,eI, ...
                                                     input',[],1);

    % take index of max probability for each frame to generate alignment
    % vector
    [p uttAlisNN] = max(output);
    uttAlisNN=uttAlisNN';

    % count accuracy
    num_frames = length(uttAlis);
    for a=1:num_frames
        nnPhone = pdf_phone_map(uttAlisNN(a));
        gtPhone = pdf_phone_map(uttAlis(a));
        numPhonesRight = numPhonesRight + (nnPhone==gtPhone);

        % increment confusion mat (ground t on rows, predicted on columns)
        confusion(gtPhone,nnPhone)=confusion(gtPhone,nnPhone)+1;

    end;

    numSonemesRight = numSonemesRight + sum(uttAlis==uttAlisNN);
    numFramesDone = numFramesDone+sum(subSizes);

end

% calculate soneme and phoneme accuracy and compare
soneAcc=numSonemesRight/length(alis)
phoneAcc=numPhonesRight/length(alis)

% visualize confusion matrix
figure;
imagesc(confusion); hold on; colorbar; 
xlabel('Predicted Phones');
ylabel('Ground Truth Phones');

% visualize confusion matrix without silence phone
figure;
imagesc(confusion(2:end,2:end)); hold on; colorbar;
xlabel('Predicted Phones');
ylabel('Ground Truth Phones');

% confusion matrix with zeroed diagonal
zerodiag = ones(num_phones)-diag(ones(num_phones,1));
figure;
imagesc(confusion.*zerodiag); hold on; colorbar;
h=gca;set(h,'XAxisLocation','Top');
xlabel('Predicted Phones');
ylabel('Ground Truth Phones');
set(h,'XTick',1:num_phones,'XTickLabel',num_to_txt);
set(h,'YTick',1:num_phones,'YTickLabel',num_to_txt);
plot(1:num_phones,1:num_phones,'w-');


%% Look at a single utterance

% choose an utterance to view
uttnum=101;
fStart = sum(utt_dat.sizes(1:uttnum-1))+1;
fEnd = fStart+utt_dat.sizes(uttnum)-1;
uttFeats = feats(fStart:fEnd,:);
uttAlis = alis(fStart:fEnd);

% forward prop utterance
[c, g, nC, nE, ceC, wC, output] = spNetCostSlave(theta,eI,uttFeats',[],1);

% take index of max probability for each frame to generate alignment
% vector
[p uttAlisNN] = max(output);
uttAlisNN=uttAlisNN';

% map both from pdf id to phone
num_frames = length(uttAlis);
uttPhones = zeros(1,num_frames);
uttPhonesNN_mass = zeros(num_phones,num_frames);
uttPhonesNN_pred = zeros(num_phones,num_frames);
nnPhoneStr = [];
gtPhoneStr = [];
gtPhonePrev = -1;
nnPhonePrev = -1;

for a=1:num_frames
    for j=1:num_pdfs
        p=pdf_phone_map(j);
        uttPhonesNN_mass(p,a)=uttPhonesNN_mass(p,a)+output(j,a);
    end

    gtPhone = pdf_phone_map(uttAlis(a));
    nnPhone = pdf_phone_map(uttAlisNN(a));

    uttPhones(a) = gtPhone;
    uttPhonesNN_pred(nnPhone,a) = 1;
    
    if gtPhonePrev ~= gtPhone
        gtPhoneStr = [gtPhoneStr num_to_txt{gtPhone} '-'];
    end;

    if nnPhonePrev ~= nnPhone
        nnPhoneStr = [nnPhoneStr num_to_txt{nnPhone} '-'];
    end
    
    gtPhonePrev = gtPhone;
    nnPhonePrev = nnPhone;

end;

gtPhoneStr=gtPhoneStr(1:end-1);
nnPhoneStr=nnPhoneStr(1:end-1);

% compare predicted phones to ground truth
figure;
imagesc(uttPhonesNN_pred); hold on;colorbar;
plot(1:num_frames,uttPhones,'k.');
title({['GT Phone Str: ' gtPhoneStr],['NN Phone Str: ' nnPhoneStr]});
xlabel('Frames of utterance');
ylabel('Phones');
set(gca,'YTick',1:num_phones,'YTickLabel',num_to_txt);

% compare mass of predicted phones to ground truth
figure;
imagesc(uttPhonesNN_mass); hold on;colorbar;
plot(1:num_frames,uttPhones,'k.');
title({['Ground Truth Phone Str: ' gtPhoneStr],['NN Phone Str: ' nnPhoneStr]});
xlabel('Frames of utterance');
ylabel('Phones');
set(gca,'YTick',1:num_phones,'YTickLabel',num_to_txt);





