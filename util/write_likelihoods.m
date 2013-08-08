function [] = write_likelihoods(kal_root,dat_in,dat_out,num_files,nn_model)
% Takes as input , location of Kaldi source, a data directory 
% with NN input data, a data directory to write loglikelihoods,
% a neural net model to use for forward propagation. Data will be
% written in kaldi readable binary format to file
% 'loglikelihoods.ark' in data_out


% Inputs:
% 
% kal_root :  root of Kaldi source
% dat_in   :  directory where data to forward prop resides
% dat_out  :  where the binary likelihoods will be dumped
% nn_model :  info needed to call forward propagate (code not yet written)


%%Setup

%add path to forward prop code
addpath ../simple-hybrid;

%load network params
load(nn_model,'eI','theta');
eI.useGpu = 0; %don't use the gpu as likely too much memory

disp(eI);


%Load priors from ali_train_pdf.counts
prior_file = ['kaldi-trunk/egs/swbd/s5/exp/' ...
              'nn_data_fmllr_train_nn/ali_train_pdf.counts'];
priors = load([kal_root prior_file]);

%Take log of inverse priors and scale
prior_scale = 1;  %Use to change weight of NN vs Priors
priors = -prior_scale*log(priors./sum(priors));

%Replace any inf values with max prior value
priors(find(priors==inf)) = -inf;
maxp = max(priors);
priors(find(priors==-inf)) = maxp;

numStates = size(priors,2); %Number of HMM states

%File where log likelihoods are written
ll_format = [dat_out 'loglikelihoods_%d.ark'];

%%Forward prop and write likelihoods to output
%loop over all files in data dir
for fn=1:num_files

    ll_out = sprintf(ll_format,fn);

    %Load features and data (keys and utt sizes)
    [feats utt_dat] = load_kaldi_data(dat_in,fn,eI.inputDim);
    assert(size(feats,1)==sum(utt_dat.sizes));
    numUtts = length(utt_dat.keys); %Number of utterances


  
    %open log likelihood file to write
    fid = fopen(ll_out,'w');

    chunkSize = 100; %Size of utterance chunks to forward prop at a time
    numChunks = ceil(numUtts/chunkSize);
    numFramesDone = 0; %Number of total frames written
    numUttsDone = 0; %Number of utterances written



    for i=1:numChunks

        %Get subset of keys and subset of sizes to forward prop and write
        if i==numChunks
            subKeys=utt_dat.keys((i-1)*chunkSize+1:end);
            subSizes=utt_dat.sizes((i-1)*chunkSize+1:end,:);
        else
            subKeys=utt_dat.keys((i-1)*chunkSize+1:i*chunkSize);
            subSizes=utt_dat.sizes((i-1)*chunkSize+1:i*chunkSize);
        end

        input = feats(numFramesDone+1:numFramesDone+sum(subSizes),:);    

  

        %%%%%%%%%%%%% Testing Setups
        % Oracle setup for unit testing
        %inputalis = alis(numFramesDone+1:numFramesDone+sum(subSizes))+1;
        %output = 1e-6*ones(size(input,1),size(priors,2));
        %output(sub2ind(size(output),(1:size(input,1))',inputalis))=.999;
        
        % Random setup for unit testing
        %rand('seed',0);
        %output = rand(size(input,1),size(priors,2)); %filler data
        %bsxfun(@rdivide,output,sum(output,2)); %normalize
        %output=output';
        %%%%%%%%%%%%%

        %Forward prop data in cost function
        [c, g, nC, nE, ceC, wC, output] = spNetCostSlave(theta,eI,input',[],1);

        %take log of forward propped dat and add log inverse priors
        output = bsxfun(@plus,log(output'),priors);

        %Write each utterance separately so we can write as key value pairs
        numFramesWrit = 0;
        for u=1:length(subKeys)
            uttSize = subSizes(u);
            FLOATSIZE=4;
            %write each key with corresponding nnet value
            fprintf(fid,'%s ',subKeys{u}); % write key
            fprintf(fid,'%cBFM ',char(0)); % write Kaldi header
            fwrite(fid,FLOATSIZE,'integer*1'); %write size of float as 1
                                               %byte int
            fwrite(fid,uttSize,'int'); % write number rows
            fwrite(fid,FLOATSIZE,'integer*1'); %write size of float as 1
                                               %byte int
            fwrite(fid,numStates,'int');  % write number cols

            % write full utterance (have to transpose as fwrite is column order
            fwrite(fid,output(numFramesWrit+1:numFramesWrit+uttSize,:)', ...
                   'float'); 

            %% Commented out writing text version, replaced with writing binary version
            %% which is significantly faster
% $$$       fprintf(fid,'%s  [\n  ',subKeys{u});
% $$$     
% $$$       %Write each row of mat separately
% $$$       for j=1:uttSize
% $$$         fprintf(fid,'%g ',output(numFramesWrit+j,:));
% $$$         if j==uttSize
% $$$           fprintf(fid,']\n');
% $$$         else
% $$$           fprintf(fid,'\n');
% $$$         end
% $$$       end
            numFramesWrit = numFramesWrit+uttSize;
        end
        numUttsDone = numUttsDone+length(subKeys);
        numFramesDone = numFramesDone+sum(subSizes);
    end
    
    fprintf('%d of %d files written\n',fn,num_files);

    %close log likelihood file
    fclose(fid);
end

end
